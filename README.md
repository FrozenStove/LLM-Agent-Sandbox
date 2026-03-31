# Prior Authorization API

A FastAPI backend that uses LLMs to automate prior authorization form completion for pharmacy workflows. Given a patient record and a set of prior authorization questions, it returns structured answers with explanations and confidence scores.

## Features

- **`/answers`** — Batch endpoint: answers all questions in a single LLM call
- **`/answers/stream`** — Streaming endpoint: returns answers via Server-Sent Events as they are generated
- **`/answers/actor-critic`** — Two independent LLM calls (actor + critic); disagrees with itself to catch low-confidence answers
- **`visible_if` conditional logic** — Two-pass approach that only sends relevant follow-up questions to the model
- **Eval pipeline** — Benchmarks accuracy and ECE (Expected Calibration Error) against labeled cases
- **Frontend** — Single-file HTML UI for reviewing and editing answers

## Tech Stack

- Python 3.10+, FastAPI, Pydantic v2
- OpenAI `gpt-4o` with structured output parsing (`beta.chat.completions.parse`)
- Logfire for observability
- pytest-asyncio, httpx for testing
- uv for package management

## Setup

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
uv sync
uv run dev
```

## Running Tests

```bash
uv run pytest
```

## Design

See [documentation/Design Writeup.md](documentation/Design%20Writeup.md) for design decisions, tradeoffs, and known limitations.
