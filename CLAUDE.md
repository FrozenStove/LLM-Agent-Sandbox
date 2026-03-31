# AI Agent Guidelines

This document provides context and guidelines for AI assistants working on this codebase.

## Project Overview

This is a FastAPI backend for a prior authorization automation system. The `/answers` endpoint uses an LLM to generate answers to prior authorization questions based on patient data.

## Technology Stack

- **Framework**: FastAPI with async support
- **Package Manager**: uv
- **Python Version**: 3.10+
- **Validation**: Pydantic v2
- **LLM Integration**: OpenAI (primary), Anthropic or Google supported

## Code Generation Guidelines

### Error Handling

Use try-except blocks around LLM calls:

```python
try:
    response = await client.chat.completions.create(...)
except Exception as e:
    logger.error(f"LLM call failed: {e}")
    raise HTTPException(status_code=500, detail="Failed to generate answers")
```

### Environment Variables

Required environment variables:
- `OPENAI_API_KEY` - For OpenAI integration
- `ANTHROPIC_API_KEY` - For Claude integration  
- `GOOGLE_API_KEY` - For Gemini integration

Optional:
- `LOGFIRE_TOKEN` - For Pydantic Logfire observability
- `LLM_MODEL` - Model override (default: gpt-4o)

## Testing Guidelines

Write integration tests that verify the endpoint behavior:

```python
def test_answers_returns_correct_structure():
    response = client.post("/answers", json=test_payload)
    assert response.status_code == 200
    assert "answers" in response.json()
```
