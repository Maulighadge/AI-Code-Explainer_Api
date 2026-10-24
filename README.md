# AI Code Explainer — Backend API 🔧

FastAPI-based REST API powered by **Google Gemini 1.5 Flash** for intelligent code analysis.

## Features

| Endpoint | Description |
|---|---|
| `POST /api/explain` | Line-by-line explanation + summary |
| `POST /api/complexity` | Time & space complexity analysis |
| `POST /api/improvements` | Code improvement suggestions + refactored code |
| `POST /api/tests` | Auto-generated unit tests |
| `POST /api/analyze-all` | All four analyses in one call |

## Supported Languages

🐍 Python · ⚡ JavaScript · 🐹 Go

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Maulighadge/AI-Code-Explainer_Api.git
cd AI-Code-Explainer_Api
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env and paste your Gemini API key
```

Get your free API key at: https://aistudio.google.com/app/apikey

### 5. Run the server
```bash
uvicorn main:app --reload
```

Server starts at: `http://localhost:8000`  
Interactive docs at: `http://localhost:8000/docs`

## Tech Stack

- **FastAPI** — High-performance async API framework
- **Google Gemini 1.5 Flash** — LLM for code analysis
- **Pydantic v2** — Request/response validation
- **Uvicorn** — ASGI server

## Project Structure

```
AI-Code-Explainer_Api/
├── main.py          # FastAPI app, routes, prompts
├── requirements.txt # Python dependencies
├── .env.example     # Environment template
└── README.md
```

## License

MIT
