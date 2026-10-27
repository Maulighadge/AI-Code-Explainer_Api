import os
import json
import re
import asyncio
import hashlib
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import (
    ResourceExhausted,
    PermissionDenied,
    InvalidArgument,
    DeadlineExceeded,
    ServiceUnavailable,
)

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_code_explainer")

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set. Please check your .env file.")

genai.configure(api_key=api_key)

# gemini-2.0-flash is faster than 1.5-flash with comparable quality
model = genai.GenerativeModel(
    "gemini-2.0-flash",
    generation_config=genai.GenerationConfig(
        temperature=0.2,        # lower temp = faster, more deterministic
        max_output_tokens=2048, # cap tokens to avoid runaway responses
    ),
)

app = FastAPI(
    title="AI Code Explainer API",
    description="Explain, analyze, and improve code using Google Gemini AI",
    version="2.0.0",
)

# CORS - allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Exception Handlers ──────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return clean, user-friendly validation error messages."""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error.get("loc", []) if loc != "body")
        errors.append(f"{field}: {error.get('msg')}" if field else error.get('msg', 'Invalid input'))
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error: " + "; ".join(errors), "errors": errors},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Fallback handler to prevent leaking raw tracebacks to clients."""
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."},
    )

# ── Request / Response Models ──────────────────────────────────────────────────

SUPPORTED_LANGUAGES = {"python", "javascript", "go"}
MAX_CODE_CHARS = 20000

class CodeRequest(BaseModel):
    code: str
    language: str  # "python" | "javascript" | "go"

    @field_validator("code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Code snippet cannot be empty.")
        if len(trimmed) > MAX_CODE_CHARS:
            raise ValueError(f"Code snippet exceeds maximum allowed size ({MAX_CODE_CHARS} characters).")
        return trimmed

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        lang = v.strip().lower()
        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{v}'. Supported options: {', '.join(sorted(SUPPORTED_LANGUAGES))}")
        return lang

class ExplainResponse(BaseModel):
    line_by_line: str
    summary: str

class AnalysisResponse(BaseModel):
    time_complexity: str
    space_complexity: str
    explanation: str

class ImprovementsResponse(BaseModel):
    improvements: str
    improved_code: str

class TestsResponse(BaseModel):
    unit_tests: str
    explanation: str

class FullAnalysisResponse(BaseModel):
    explanation: ExplainResponse
    complexity: AnalysisResponse
    improvements: ImprovementsResponse
    tests: TestsResponse


# ── In-memory LRU response cache ──────────────────────────────────────────────

_cache: dict[str, str] = {}
_CACHE_MAX = 128
_cache_keys: list[str] = []  # insertion-order tracking for simple LRU eviction

def _cache_key(task: str, code: str, language: str) -> str:
    return hashlib.sha256(f"{task}:{language}:{code}".encode()).hexdigest()

def _cache_get(key: str) -> str | None:
    return _cache.get(key)

def _cache_set(key: str, value: str) -> None:
    if key in _cache:
        return
    if len(_cache_keys) >= _CACHE_MAX:
        oldest = _cache_keys.pop(0)
        _cache.pop(oldest, None)
    _cache[key] = value
    _cache_keys.append(key)


# ── Async Gemini caller ────────────────────────────────────────────────────────

def map_gemini_exception(e: Exception) -> HTTPException:
    """Map specific Gemini API errors to proper HTTP status codes."""
    if isinstance(e, ResourceExhausted):
        logger.warning("Gemini API rate limit or quota exceeded: %s", e)
        return HTTPException(
            status_code=429,
            detail="Rate limit or quota exceeded with the AI service. Please wait a moment and try again.",
        )
    if isinstance(e, PermissionDenied):
        logger.error("Gemini API authentication failed: %s", e)
        return HTTPException(
            status_code=401,
            detail="Invalid or unauthorized Gemini API key. Please check your credentials.",
        )
    if isinstance(e, InvalidArgument):
        logger.warning("Invalid argument passed to Gemini: %s", e)
        return HTTPException(
            status_code=400,
            detail=f"The AI service could not process the request: {str(e)}",
        )
    if isinstance(e, DeadlineExceeded):
        logger.warning("Gemini API request timed out: %s", e)
        return HTTPException(
            status_code=504,
            detail="AI service request timed out. Please try again with shorter code.",
        )
    if isinstance(e, ServiceUnavailable):
        logger.warning("Gemini API unavailable: %s", e)
        return HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable. Please try again shortly.",
        )

    # General / fallback error handling
    err_str = str(e).lower()
    if "quota" in err_str or "rate limit" in err_str or "429" in err_str:
        return HTTPException(
            status_code=429,
            detail="AI service quota exceeded. Please try again shortly.",
        )
    if "api key" in err_str or "unauthorized" in err_str:
        return HTTPException(
            status_code=401,
            detail="Authentication failed with AI service.",
        )

    logger.error("Gemini API error: %s", e)
    return HTTPException(status_code=502, detail=f"Gemini API error: {str(e)}")


async def call_gemini_async(prompt: str, cache_key: str | None = None) -> str:
    """
    Non-blocking Gemini call.
    Runs the synchronous SDK in a thread pool so the event loop stays free,
    enabling true parallelism via asyncio.gather in /api/analyze-all.
    Repeated identical requests are served from the in-memory cache.
    """
    if cache_key:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    try:
        response = await asyncio.to_thread(model.generate_content, prompt)
        if not response or not response.text:
            raise HTTPException(
                status_code=502,
                detail="Empty response received from the AI model.",
            )
        text = response.text
    except HTTPException:
        raise
    except Exception as e:
        raise map_gemini_exception(e)

    if cache_key:
        _cache_set(cache_key, text)
    return text


# ── Prompts ────────────────────────────────────────────────────────────────────

def explain_prompt(code: str, language: str) -> str:
    return f"""You are an expert {language} developer and educator.

Analyze the following {language} code and provide:
1. A line-by-line explanation - for each meaningful line or block, explain what it does in plain English.
   Format each line explanation as: Line X: <explanation>
2. A summary paragraph (3-5 sentences) explaining the overall purpose and flow of the code.

Code:
```{language}
{code}
```

Return your response in this exact JSON format (no markdown fences around the JSON itself):
{{
  "line_by_line": "<full line-by-line explanation as a single formatted string with newlines>",
  "summary": "<summary paragraph>"
}}"""


def complexity_prompt(code: str, language: str) -> str:
    return f"""You are an expert algorithm analyst.

Analyze the time and space complexity of the following {language} code.

Code:
```{language}
{code}
```

Return your response in this exact JSON format (no markdown fences around the JSON itself):
{{
  "time_complexity": "<Big-O notation, e.g. O(n log n)>",
  "space_complexity": "<Big-O notation, e.g. O(n)>",
  "explanation": "<2-4 sentences explaining why these complexities apply>"
}}"""


def improvements_prompt(code: str, language: str) -> str:
    return f"""You are a senior {language} developer performing a code review.

Review the following {language} code and provide:
1. Suggested improvements as a numbered list covering: readability, performance, error handling, best practices, and security.
2. An improved version of the code incorporating your suggestions.

Code:
```{language}
{code}
```

Return your response in this exact JSON format (no markdown fences around the JSON itself):
{{
  "improvements": "<numbered list of improvements as a formatted string>",
  "improved_code": "<the improved code as a plain string>"
}}"""


def tests_prompt(code: str, language: str) -> str:
    return f"""You are an expert {language} developer specializing in test-driven development.

Generate comprehensive unit tests for the following {language} code.
- Cover happy paths, edge cases, and error cases.
- Use the standard testing framework for {language} (pytest for Python, Vitest for JavaScript, testing package for Go).
- Add brief comments explaining what each test verifies.

Code:
```{language}
{code}
```

Return your response in this exact JSON format (no markdown fences around the JSON itself):
{{
  "unit_tests": "<the complete test code as a plain string>",
  "explanation": "<1-3 sentences describing the testing strategy>"
}}"""


# ── Utility: parse Gemini JSON ─────────────────────────────────────────────────

def parse_json_response(raw: str, task_name: str = "analysis") -> dict:
    """Extract and parse JSON from a Gemini response that might have markdown fences."""
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("Failed to decode JSON from AI response for %s: %s | Raw snippet: %s", task_name, e, raw[:250])
        raise HTTPException(
            status_code=502,
            detail=f"The AI returned an invalid response format for {task_name}. Please try running it again.",
        )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "AI Code Explainer API is running", "version": "2.0.0"}


@app.get("/api/cache/stats")
async def cache_stats():
    """Return current in-memory cache statistics."""
    return {"cached_entries": len(_cache), "max_entries": _CACHE_MAX}


@app.post("/api/explain", response_model=ExplainResponse)
async def explain_code(req: CodeRequest):
    """Explain the given code line by line and provide a summary."""
    key = _cache_key("explain", req.code, req.language)
    raw = await call_gemini_async(explain_prompt(req.code, req.language), cache_key=key)
    return ExplainResponse(**parse_json_response(raw, "explain"))


@app.post("/api/complexity", response_model=AnalysisResponse)
async def analyze_complexity(req: CodeRequest):
    """Analyze the time and space complexity of the code."""
    key = _cache_key("complexity", req.code, req.language)
    raw = await call_gemini_async(complexity_prompt(req.code, req.language), cache_key=key)
    return AnalysisResponse(**parse_json_response(raw, "complexity"))


@app.post("/api/improvements", response_model=ImprovementsResponse)
async def suggest_improvements(req: CodeRequest):
    """Suggest code improvements and return an improved version."""
    key = _cache_key("improvements", req.code, req.language)
    raw = await call_gemini_async(improvements_prompt(req.code, req.language), cache_key=key)
    return ImprovementsResponse(**parse_json_response(raw, "improvements"))


@app.post("/api/tests", response_model=TestsResponse)
async def generate_tests(req: CodeRequest):
    """Generate unit tests for the given code."""
    key = _cache_key("tests", req.code, req.language)
    raw = await call_gemini_async(tests_prompt(req.code, req.language), cache_key=key)
    return TestsResponse(**parse_json_response(raw, "tests"))


@app.post("/api/analyze-all", response_model=FullAnalysisResponse)
async def analyze_all(req: CodeRequest):
    """
    Run all 4 analyses (explain + complexity + improvements + tests) in PARALLEL.
    Previously sequential (~4x slower). Now uses asyncio.gather for concurrent
    Gemini calls, cutting total latency to roughly the slowest single call.
    """
    explain_key    = _cache_key("explain",      req.code, req.language)
    complexity_key = _cache_key("complexity",   req.code, req.language)
    improve_key    = _cache_key("improvements", req.code, req.language)
    tests_key      = _cache_key("tests",        req.code, req.language)

    results = await asyncio.gather(
        call_gemini_async(explain_prompt(req.code, req.language),      cache_key=explain_key),
        call_gemini_async(complexity_prompt(req.code, req.language),   cache_key=complexity_key),
        call_gemini_async(improvements_prompt(req.code, req.language), cache_key=improve_key),
        call_gemini_async(tests_prompt(req.code, req.language),        cache_key=tests_key),
        return_exceptions=True,
    )

    labels = ["explanation", "complexity", "improvements", "tests"]
    errors = [
        f"{labels[i]}: {str(res.detail if isinstance(res, HTTPException) else res)}"
        for i, res in enumerate(results)
        if isinstance(res, Exception)
    ]

    if errors:
        first_http_exc = next((r for r in results if isinstance(r, HTTPException)), None)
        status_code = first_http_exc.status_code if first_http_exc else 502
        raise HTTPException(
            status_code=status_code,
            detail=f"Analysis failed: {'; '.join(errors)}",
        )

    explain_raw, complexity_raw, improvements_raw, tests_raw = results

    return FullAnalysisResponse(
        explanation =ExplainResponse(**parse_json_response(explain_raw, "explain")),
        complexity  =AnalysisResponse(**parse_json_response(complexity_raw, "complexity")),
        improvements=ImprovementsResponse(**parse_json_response(improvements_raw, "improvements")),
        tests       =TestsResponse(**parse_json_response(tests_raw, "tests")),
    )
