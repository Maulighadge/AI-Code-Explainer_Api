import os
import json
import re
import asyncio
import hashlib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

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


# ── Request / Response Models ──────────────────────────────────────────────────

class CodeRequest(BaseModel):
    code: str
    language: str  # "python" | "javascript" | "go"

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
        text = response.text
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {str(e)}")

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

def parse_json_response(raw: str) -> dict:
    """Extract and parse JSON from a Gemini response that might have markdown fences."""
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse AI response: {str(e)}\nRaw: {raw[:300]}"
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
    return ExplainResponse(**parse_json_response(raw))


@app.post("/api/complexity", response_model=AnalysisResponse)
async def analyze_complexity(req: CodeRequest):
    """Analyze the time and space complexity of the code."""
    key = _cache_key("complexity", req.code, req.language)
    raw = await call_gemini_async(complexity_prompt(req.code, req.language), cache_key=key)
    return AnalysisResponse(**parse_json_response(raw))


@app.post("/api/improvements", response_model=ImprovementsResponse)
async def suggest_improvements(req: CodeRequest):
    """Suggest code improvements and return an improved version."""
    key = _cache_key("improvements", req.code, req.language)
    raw = await call_gemini_async(improvements_prompt(req.code, req.language), cache_key=key)
    return ImprovementsResponse(**parse_json_response(raw))


@app.post("/api/tests", response_model=TestsResponse)
async def generate_tests(req: CodeRequest):
    """Generate unit tests for the given code."""
    key = _cache_key("tests", req.code, req.language)
    raw = await call_gemini_async(tests_prompt(req.code, req.language), cache_key=key)
    return TestsResponse(**parse_json_response(raw))


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

    (explain_raw, complexity_raw, improvements_raw, tests_raw) = await asyncio.gather(
        call_gemini_async(explain_prompt(req.code, req.language),      cache_key=explain_key),
        call_gemini_async(complexity_prompt(req.code, req.language),   cache_key=complexity_key),
        call_gemini_async(improvements_prompt(req.code, req.language), cache_key=improve_key),
        call_gemini_async(tests_prompt(req.code, req.language),        cache_key=tests_key),
    )

    return FullAnalysisResponse(
        explanation =ExplainResponse(**parse_json_response(explain_raw)),
        complexity  =AnalysisResponse(**parse_json_response(complexity_raw)),
        improvements=ImprovementsResponse(**parse_json_response(improvements_raw)),
        tests       =TestsResponse(**parse_json_response(tests_raw)),
    )
