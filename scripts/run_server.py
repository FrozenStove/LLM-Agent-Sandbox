"""Run the FastAPI dev server with hot reload.

Usage:
    uv run dev
    uv run run
"""

import uvicorn


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
