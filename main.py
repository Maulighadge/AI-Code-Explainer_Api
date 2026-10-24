import os
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
model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI(
    title="AI Code Explainer API",
    description="Explain, analyze, and improve code using Google Gemini AI",
    version="1.0.0",
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


# ── Helper ─────────────────────────────────────────────────────────────────────

def call_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return the text response."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {str(e)}")


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

import json
import re

def parse_json_response(raw: str) -> dict:
    """Extract and parse JSON from a Gemini response that might have markdown fences."""
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}\nRaw: {raw[:300]}")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "AI Code Explainer API is running", "version": "1.0.0"}


@app.post("/api/explain", response_model=ExplainResponse)
async def explain_code(req: CodeRequest):
    """Explain the given code line by line and provide a summary."""
    raw = call_gemini(explain_prompt(req.code, req.language))
    data = parse_json_response(raw)
    return ExplainResponse(**data)


@app.post("/api/complexity", response_model=AnalysisResponse)
async def analyze_complexity(req: CodeRequest):
    """Analyze the time and space complexity of the code."""
    raw = call_gemini(complexity_prompt(req.code, req.language))
    data = parse_json_response(raw)
    return AnalysisResponse(**data)


@app.post("/api/improvements", response_model=ImprovementsResponse)
async def suggest_improvements(req: CodeRequest):
    """Suggest code improvements and return an improved version."""
    raw = call_gemini(improvements_prompt(req.code, req.language))
    data = parse_json_response(raw)
    return ImprovementsResponse(**data)


@app.post("/api/tests", response_model=TestsResponse)
async def generate_tests(req: CodeRequest):
    """Generate unit tests for the given code."""
    raw = call_gemini(tests_prompt(req.code, req.language))
    data = parse_json_response(raw)
    return TestsResponse(**data)


@app.post("/api/analyze-all", response_model=FullAnalysisResponse)
async def analyze_all(req: CodeRequest):
    """Run all analyses (explain + complexity + improvements + tests) in sequence."""
    explain_raw = call_gemini(explain_prompt(req.code, req.language))
    explain_data = parse_json_response(explain_raw)

    complexity_raw = call_gemini(complexity_prompt(req.code, req.language))
    complexity_data = parse_json_response(complexity_raw)

    improvements_raw = call_gemini(improvements_prompt(req.code, req.language))
    improvements_data = parse_json_response(improvements_raw)

    tests_raw = call_gemini(tests_prompt(req.code, req.language))
    tests_data = parse_json_response(tests_raw)

    return FullAnalysisResponse(
        explanation=ExplainResponse(**explain_data),
        complexity=AnalysisResponse(**complexity_data),
        improvements=ImprovementsResponse(**improvements_data),
        tests=TestsResponse(**tests_data),
    )
