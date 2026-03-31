# Design Write Up

My main scripting language is JavaScript/TypeScript, so it took me a bit of extra time to translate my engineering knowledge and best practices into Python.

My first step was to familiarize myself with the existing codebase and check for any existing AI model rules and conventions. I then inserted my own development guidelines and parsed through the existing code to verify quality and unify standards. It's important to keep a tight leash and strict bounds with AI coding agents, as they can often go off the rails and start developing duplicate or bad code. From there I wrote up a detailed implementation plan covering all of the significant features and sections before writing any code.

## Design Choices

**Structured output via `beta.chat.completions.parse`**
In order to get a properly enforced JSON schema back from the model, I used `beta.chat.completions.parse` on the newer `gpt-4o` model rather than standard completions. Standard completions return free-form text — even if you ask for JSON, the model can include markdown formatting, trailing commas, or truncated output that breaks parsing. `beta.chat.completions.parse` takes a Pydantic model directly and guarantees the response matches that schema, so there is no manual parsing needed.

**Two-pass visible_if approach**
Conditional questions can't be sent to the LLM before knowing whether they're relevant. The backend answers all unconditional questions first, uses those answers to determine which conditional follow-up questions are now applicable, then makes a second LLM call with only the relevant ones. This keeps the model focused on questions that actually apply to the patient rather than answering questions that shouldn't be asked.

**Batch vs per-question LLM calls**
The `/answers` endpoint sends all questions in a single call to minimize overhead and keep answers consistent with each other. The `/answers/stream` endpoint sends one question at a time so answers can be returned to the frontend as they arrive, rather than waiting for the full batch. The tradeoff is significantly more token usage for streaming, but the UX improvement is meaningful for long question sets.

**Actor-Critic system**
I built a second endpoint `/answers/actor-critic` that runs two independent LLM calls. The actor answers all questions normally. The critic is given the same patient data and questions but has never seen the actor's output, so it forms its own answer independently. The two responses are then compared in Python — if they disagree and the critic's confidence is above 0.7, the critic's answer is used. All three values (actor answer, final answer, critic critique) are returned so a reviewer can see exactly what happened.

My initial attempt at the critic was poor but functional — I had it reading the actor's answer, which caused it to anchor on that response rather than forming its own judgment. Refactoring it to run completely independently was the key fix.

**Eval pipeline**
I built an evaluation pipeline in `scripts/eval_pipeline.py` that runs the actor-critic system against a set of labeled cases in `sample_data/labeled_cases.json`. Each case has a patient, questions, and pre-written correct answers. The pipeline compares the model's output against those answers and computes ECE (Expected Calibration Error) — a metric that measures whether confidence scores are honest. If the model says 0.9 confidence on 10 questions, it should actually be correct on roughly 9 of them. A lower ECE means the confidence scores are trustworthy. The labeled cases don't get fed into the model — they're purely a benchmark used after the fact to measure accuracy and catch regressions when the prompt or model changes. In the future this should be validated against real patient data with appropriate consent rather than the synthetic cases currently in the file.

**Frontend**
I added a basic frontend as a single HTML file with inline JavaScript and Tailwind for CSS. This was developed with an AI agent and is intended only for testing the backend — it is not production ready. Keeping it as a single file means no build step, no dependencies, and no deployment complexity.

**Testing**
I switched over to `pytest-asyncio` for async test support. The test suite uses `httpx.AsyncClient` with `ASGITransport` to wire the client directly to the FastAPI app. This allows the tests to run quickly and LLM calls can be mocked without a testing API key. `conftest.py` sets up this transport once and shares it across all tests via a fixture.

## Known Bugs

1. **Nested visible_if** — the backend only processes two rounds of questions, so it only resolves one layer of conditional visibility. Any question whose `visible_if` depends on another conditional question will never be answered. The fix would be a while loop that keeps checking for newly visible questions after each round, with a hard limit to prevent infinite loops.

2. **Medication anchoring in the system prompt** — the system prompt instructs the LLM to answer questions about the medication listed in the patient's `prescription` field, not any prior medications mentioned in the visit notes. In practice this doesn't always hold. For example, the question "How long has the patient been on the requested medication?" returned "14 months" with the reasoning "The visit notes state that Mr. Chaney has been on Wegovy for 14 months." The questionnaire was for Zepbound, so the correct answer is 0 months. The prompt instruction is not consistently overriding the model's tendency to answer based on what's prominent in the notes.

3. **visible_if parser** — the parser only supports AND conditions via a simple string split on `" and "`. OR conditions are not supported at all. It also requires strict string formatting — any capitalisation difference, extra whitespace, or missing curly braces causes a silent failure where the question is hidden with no log or warning. There are likely existing packages that could handle this parsing more robustly.

4. **stream_answers is synchronous** — each question in the stream endpoint is sent to the LLM one at a time and awaited before the next one starts. This compounds the nested visible_if problem because conditional questions can't be resolved mid-stream. Ideally questions would be sent concurrently where possible, with visibility re-evaluated after each answer resolves.

## What I'd Build Next

1. More comprehensive logging through Logfire — currently only basic request logging is set up. Better observability would help track which question sets are consuming the most tokens and catch accuracy regressions in production.

2. A database to persist responses over time. This would let us plot average confidence and track accuracy trends, and corrections made by reviewers could feed back into the labeled cases dataset.

3. A proper React frontend with TypeScript. The single HTML file is fine for testing but a real reviewer UI would need proper state management, type safety on the API contract, and component structure that can be maintained.

4. A second LLM provider for the critic. Using the same model for both actor and critic means they may be biased toward the same answer. Using a different provider would make the critic more genuinely independent.

## AI Tooling

I used a combination of Cursor's auto agent and Claude Code for most of the implementation. Since I am not as familiar with Python, the AI agents are significantly faster at producing properly typed and syntactic Python code than I am. I used AI to develop a comprehensive implementation plan and reviewed each step, keeping everything well documented through incremental commits with meaningful commit messages.
