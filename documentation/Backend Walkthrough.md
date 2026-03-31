# Backend Logic Walkthrough

A step-by-step breakdown of how a request travels through the backend — from raw JSON input to a structured answer response. Includes the standard `/answers` flow, the streaming flow, and the actor-critic flow.

---

## 1. What comes in: the request body

Every request to `/answers`, `/answers/stream`, or `/answers/actor-critic` starts with the same JSON shape — a patient record and a list of questions.

```json
{
  "patient": {
    "first_name": "John",
    "last_name": "Doe",
    "date_of_birth": "1970-01-01",
    "gender": "Male",
    "prescription": {
      "medication": "Zepbound",
      "dosage": "5 mg",
      "frequency": "once weekly",
      "duration": "ongoing"
    },
    "visit_notes": ["Patient presents for follow-up regarding weight management therapy."]
  },
  "question_set": {
    "name": "Prior Authorization Questions",
    "questions": [
      { "type": "text",    "key": "patient_bmi",              "content": "What is the patient's BMI?" },
      { "type": "boolean", "key": "tried_lifestyle_modifications", "content": "Has the patient tried lifestyle modifications?" },
      { "type": "boolean", "key": "continuation",             "content": "Is this a continuation of therapy?" },
      { "type": "text",    "key": "cont_duration",            "content": "How long has the patient been on the medication?",
        "visible_if": "{continuation} = true" }
    ]
  }
}
```

**Notice** `cont_duration` has a `visible_if` field. That question should only be answered if `continuation` comes back `true`. This drives the two-pass logic below.

FastAPI validates this body against the `AnswerInput` Pydantic model before the route function ever runs:

```python
# app/models.py
class AnswerInput(BaseModel):
    patient: Patient
    question_set: QuestionSet
```

If any field is missing or the wrong type, FastAPI returns a `422` automatically — no extra code needed.

---

## 2. Standard `/answers` flow

```
POST /answers
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Route: get_answers(data: AnswerInput)           │
│  app/main.py : 83                               │
│                                                 │
│  Split questions into two buckets:              │
│   unconditional  = questions where              │
│                    visible_if is None           │
│   conditional    = questions where              │
│                    visible_if is NOT None       │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  PASS 1 — answer_questions(patient,             │
│           unconditional questions)              │
│  app/llm.py : 76                                │
│                                                 │
│  Serializes patient + questions to JSON         │
│  Sends ONE call to OpenAI with ALL questions    │
│  Gets back a structured list of answers         │
└───────────────┬─────────────────────────────────┘
                │
                │  first_answers = [Answer(...), Answer(...), ...]
                ▼
┌─────────────────────────────────────────────────┐
│  Build answer map for visibility check:         │
│  current_answers = {                            │
│    "patient_bmi":   "35.2 kg/m²",              │
│    "continuation":  "true",                     │
│    ...                                          │
│  }                                              │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  filter_visible_questions(conditional,          │
│                           current_answers)      │
│  app/visibility.py : 39                         │
│                                                 │
│  Checks each conditional question's             │
│  visible_if string against current_answers.     │
│  "{continuation} = true"  →  continuation       │
│  answer is "true"  →  INCLUDE cont_duration     │
└───────────────┬─────────────────────────────────┘
                │
                │  visible_conditional = [cont_duration, ...]
                ▼
┌─────────────────────────────────────────────────┐
│  PASS 2 — answer_questions(patient,             │
│           visible_conditional questions)        │
│  (same function, different question list)       │
└───────────────┬─────────────────────────────────┘
                │
                │  second_answers = [Answer(...), ...]
                ▼
┌─────────────────────────────────────────────────┐
│  Return AnswerOutput(                           │
│    answers = first_answers + second_answers     │
│  )                                              │
└─────────────────────────────────────────────────┘
```

### Route code (app/main.py:82–103)

```python
@app.post("/answers")
async def get_answers(data: AnswerInput) -> AnswerOutput:
    all_questions = data.question_set.questions

    # Bucket 1: always answer these
    unconditional = [q for q in all_questions if q.visible_if is None]

    # PASS 1 — one LLM call for all unconditional questions
    first_answers = await answer_questions(
        data.patient,
        data.question_set.__class__(
            name=data.question_set.name, questions=unconditional
        ),
    )

    # Turn the answers into a dict so visibility.py can evaluate conditions
    current_answers = {a.question_id: a.answer for a in first_answers}

    # Bucket 2: only answer if their visible_if condition is met
    conditional = [q for q in all_questions if q.visible_if is not None]
    visible_conditional = filter_visible_questions(conditional, current_answers)

    if not visible_conditional:
        return AnswerOutput(answers=first_answers)   # early exit if nothing is conditional

    # PASS 2 — one more LLM call for the now-visible conditional questions
    second_answers = await answer_questions(
        data.patient,
        data.question_set.__class__(
            name=data.question_set.name, questions=visible_conditional
        ),
    )

    return AnswerOutput(answers=first_answers + second_answers)
```

---

## 3. Inside answer_questions — what actually goes to the LLM

```python
# app/llm.py : 76
async def answer_questions(patient: Patient, question_set: QuestionSet) -> list[Answer]:
    if not question_set.questions:
        return []   # short-circuit: nothing to ask

    # Step A: serialize patient to JSON string
    patient_json = patient.model_dump_json(indent=2)

    # Step B: format questions into a clean list
    questions_json = _format_questions(question_set.questions)

    # Step C: send to OpenAI
    response = await get_client().beta.chat.completions.parse(
        model=settings.llm_model,          # "gpt-4o" by default
        messages=[
            {"role": "system", "content": _system_prompt()},
            {
                "role": "user",
                "content": f"Patient Data:\n{patient_json}\n\nQuestions:\n{questions_json}",
            },
        ],
        response_format=LLMAnswerList,     # <-- tells OpenAI to return structured JSON
    )

    # Step D: unpack the parsed Pydantic model
    result = response.choices[0].message.parsed
    return result.answers if result else []
```

### What `_format_questions` produces (app/llm.py:66)

```python
def _format_questions(questions: list[Question]) -> str:
    return json.dumps(
        [
            {"question_id": q.key, "content": q.content, "type": q.type}
            for q in questions
        ],
        indent=2,
    )
```

It strips `visible_if` from what gets sent to the LLM — the LLM only needs to know the question ID, content, and type. The conditional logic is handled entirely in Python.

### What `response_format=LLMAnswerList` does

`beta.chat.completions.parse` is OpenAI's **structured output** API. Instead of returning free-form text, it forces the response to match the schema of a Pydantic model:

```python
# app/llm.py : 32
class LLMAnswerList(BaseModel):
    answers: list[Answer]
```

```python
# app/models.py : 18
class Answer(BaseModel):
    question_id: str
    answer: str
    explanation: str | None = None
    confidence: float | None = None
```

OpenAI guarantees the JSON will fit this shape. `response.choices[0].message.parsed` comes back as an already-validated `LLMAnswerList` instance — no manual JSON parsing needed.

### What the system prompt tells the LLM (app/llm.py:36)

```python
def _system_prompt() -> str:
    today = date.today().isoformat()
    return f"""Today's date is {today}. You are a medical prior authorization assistant. Given patient data and a list of form questions, answer each question accurately based ONLY on the provided patient information.

The prior authorization is being requested for the specific medication named in the patient's prescription field. When any question references "the requested medication", "the current medication", or "the medication", interpret it as referring exclusively to that prescription medication — not to any prior medications, trials, or medication history mentioned in the visit notes.

For each question provide:
- "question_id": the exact key from the input question
- "answer": a concise answer appropriate for the question type. For boolean questions use "true" or "false". For text questions provide a direct factual answer.
- "explanation": one sentence citing which part of the patient data supports this answer
- "confidence": a float from 0.0 to 1.0 indicating how strongly the patient data supports your answer

If the patient data does not contain enough information to answer a question, set answer to "Unable to determine", explain what data is missing, and set confidence to 0.0."""
```

Key instructions:
- Answer from the patient data **only** — no outside medical knowledge
- The medication in `prescription.medication` is the target — ignore history
- Always return `question_id` that matches the input key exactly
- `confidence` is self-reported: 1.0 = explicitly in the notes, 0.5 = inferred, 0.0 = unknown

---

## 4. visible_if parsing (app/visibility.py)

```
Input question:
  visible_if = "{continuation} = true and {cont_less_6m} = false"

                        │
                        ▼
          split on " and "
                        │
              ┌─────────┴──────────┐
              │                    │
   "{continuation} = true"   "{cont_less_6m} = false"
              │                    │
          regex match          regex match
    r"\{(\w+)\}\s*=\s*(\S+)"
              │                    │
       key="continuation"    key="cont_less_6m"
       val="true"            val="false"
              │                    │
              └─────────┬──────────┘
                        │
            Look up each key in current_answers
            current_answers["continuation"] == "true"  ✓
            current_answers["cont_less_6m"] == "false" ✓
                        │
                   ALL conditions met
                        │
                   INCLUDE this question
```

```python
# app/visibility.py : 17
def _parse_condition(condition: str) -> tuple[str, str]:
    # Takes:  "{continuation} = true"
    # Returns: ("continuation", "true")
    match = re.match(r"\{(\w+)\}\s*=\s*(\S+)", condition.strip())
    if not match:
        raise ValueError(f"Unparseable visible_if condition: {condition!r}")
    return match.group(1), match.group(2)


def _evaluate_visible_if(visible_if: str, current_answers: dict[str, str]) -> bool:
    # Split "A and B and C" into ["A", "B", "C"] — all must pass
    conditions = [c.strip() for c in visible_if.split(" and ")]
    for condition in conditions:
        key, expected_value = _parse_condition(condition)
        actual = current_answers.get(key, "")
        if actual.lower() != expected_value.lower():
            return False   # any condition failing = question stays hidden
    return True
```

---

## 5. Streaming flow (/answers/stream)

Instead of waiting for all answers, the stream endpoint yields one SSE (Server-Sent Events) event per question as soon as the LLM responds.

```
POST /answers/stream
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Route: answers_stream(data: AnswerInput)       │
│  app/main.py : 127                              │
│                                                 │
│  Splits into unconditional / conditional        │
│  Returns a StreamingResponse wrapping           │
│  _stream_with_visibility()                      │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  _stream_with_visibility() — async generator   │
│  app/main.py : 140                              │
│                                                 │
│  For each unconditional question:               │
│    → call stream_answers(patient, [question])   │
│    → yield "data: {json}\n\n" to browser        │
│    → parse the yielded event to track answers   │
│       for visibility evaluation                 │
│                                                 │
│  After unconditional questions are done:        │
│    → run filter_visible_questions               │
│    → stream answers for visible conditionals    │
└─────────────────────────────────────────────────┘
```

### stream_answers — one LLM call per question (app/llm.py:189)

```python
async def stream_answers(
    patient: Patient, questions: list[Question]
) -> AsyncGenerator[str, None]:
    patient_json = patient.model_dump_json(indent=2)

    for question in questions:              # loop: one question at a time
        question_json = json.dumps({
            "question_id": question.key,
            "content":     question.content,
            "type":        question.type,
        }, indent=2)

        response = await get_client().beta.chat.completions.parse(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _single_question_system_prompt()},
                {"role": "user",   "content": f"Patient Data:\n{patient_json}\n\nQuestion:\n{question_json}"},
            ],
            response_format=Answer,         # single Answer this time, not a list
        )
        answer = response.choices[0].message.parsed

        yield f"data: {answer.model_dump_json()}\n\n"   # SSE format: "data: {...}\n\n"
```

Each `yield` sends one event to the browser immediately. The browser gets:

```
data: {"question_id":"patient_bmi","answer":"35.2 kg/m²","explanation":"Stated in visit notes.","confidence":0.95}

data: {"question_id":"continuation","answer":"true","explanation":"Visit notes describe ongoing therapy.","confidence":0.9}

data: {"question_id":"cont_duration","answer":"6 months","explanation":"Duration field in prescription.","confidence":0.85}
```

The two key differences from batch mode:
- `response_format=Answer` (single object) vs `LLMAnswerList` (list)
- The route makes N separate LLM calls vs 1 — higher cost, but the user sees answers immediately

---

## 6. Actor-critic flow (/answers/actor-critic)

The core idea: two independent LLM calls on the same patient data. The actor answers the questions. The critic answers them again without seeing the actor's output. If they disagree AND the critic is confident enough (≥ 0.7), the critic's answer wins.

```
POST /answers/actor-critic
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Route: get_answers_actor_critic(data)          │
│  app/main.py : 107                              │
│  (same two-pass visible_if split as /answers)   │
│                                                 │
│  Calls actor_critic_answers(patient, questions) │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  actor_critic_answers()  —  app/llm.py : 157                        │
│                                                                      │
│    ┌─────────────────────┐       ┌──────────────────────────────┐   │
│    │  ACTOR              │       │  CRITIC                      │   │
│    │  answer_questions() │       │  critique_answers()          │   │
│    │  app/llm.py : 76    │       │  app/llm.py : 127            │   │
│    │                     │       │                              │   │
│    │  System prompt:     │       │  System prompt:              │   │
│    │  "Answer these      │       │  "You are an INDEPENDENT     │   │
│    │   questions from    │       │   auditor. Answer these      │   │
│    │   patient data"     │       │   questions from patient     │   │
│    │                     │       │   data. You have NOT seen    │   │
│    │  Input:             │       │   any AI-generated answers"  │   │
│    │   patient + qs      │       │                              │   │
│    │                     │       │  Input:                      │   │
│    │  Returns:           │       │   patient + qs               │   │
│    │   list[Answer]      │       │   (NO actor output)          │   │
│    │   with confidence   │       │                              │   │
│    └──────────┬──────────┘       │  Returns:                    │   │
│               │                  │   list[CritiqueResult]       │   │
│               │                  │   with independent_answer    │   │
│               │                  │   and revised_confidence     │   │
│               │                  └──────────────┬───────────────┘   │
│               │                                 │                   │
│               └──────────────┬──────────────────┘                   │
│                              │                                      │
│                              ▼                                      │
│               MERGE in Python (not in LLM context)                  │
│               For each actor answer:                                 │
│                 Look up critic's answer by question_id               │
│                 Did they disagree?                                   │
│                   YES + critic.revised_confidence >= 0.7            │
│                     → use critic's answer as final                  │
│                   YES + critic.revised_confidence < 0.7             │
│                     → keep actor's answer, flag disagreement        │
│                   NO                                                │
│                     → keep actor's answer, no flag                  │
│                                                                      │
│               Returns: list[ActorCriticAnswer]                      │
└──────────────────────────────────────────────────────────────────────┘
```

### actor_critic_answers merge logic (app/llm.py:157–186)

```python
async def actor_critic_answers(
    patient: Patient,
    question_set: QuestionSet,
) -> list[ActorCriticAnswer]:

    # CALL 1: actor answers all questions
    actor_answers = await answer_questions(patient, question_set)

    # CALL 2: critic answers the SAME questions independently
    # — critique_answers() only receives patient + questions, NOT actor_answers
    critiques = await critique_answers(patient, question_set.questions)

    # Index critiques by question_id for fast lookup
    critique_map = {c.question_id: c for c in critiques}

    merged: list[ActorCriticAnswer] = []
    for ans in actor_answers:
        c = critique_map.get(ans.question_id)

        correction = None
        final_answer = ans.answer   # start with actor's answer

        if c and c.independent_answer.lower() != ans.answer.lower():
            # They disagreed — record what the critic suggested
            correction = c.independent_answer

            if c.revised_confidence >= _CORRECTION_CONFIDENCE_THRESHOLD:  # 0.7
                # Critic is confident enough → override actor
                final_answer = c.independent_answer

        merged.append(
            ActorCriticAnswer(
                question_id=ans.question_id,
                actor_answer=ans.answer,        # what the actor said
                answer=final_answer,            # what we're going with
                explanation=ans.explanation,
                actor_confidence=ans.confidence,
                critic_confidence=c.revised_confidence if c else None,
                critique=c.critique if c else None,
                suggested_correction=correction, # set if they disagreed
            )
        )
    return merged
```

### What the critic prompt does differently (app/llm.py:111)

```python
def _critic_system_prompt() -> str:
    return f"""Today's date is {today}. You are an independent medical prior authorization auditor.
You have NOT seen any AI-generated answers. Answer each question yourself using ONLY the provided patient data.

For each question, return:
- "question_id": exact key from the input
- "independent_answer": your own answer based solely on the patient data
- "is_supported": true if patient data directly supports your answer
- "critique": one sentence citing the specific data that supports or contradicts answerability
- "revised_confidence": your calibrated confidence (0.0–1.0)

Be conservative: ambiguous or missing data should lower confidence, not raise it."""
```

The key line is **"You have NOT seen any AI-generated answers."** The actor's output is never included in the critic's message, so the critic cannot be anchored to the actor's phrasing. It forms its own answer from scratch using the same patient data.

### Why this matters

- If the **actor** answers `"true"` for `continuation` and the **critic** independently answers `"true"` → high agreement, no correction, confidence is reinforced
- If the actor answers `"true"` but the critic answers `"false"` with `revised_confidence=0.85` → correction applied, `answer` becomes `"false"`, `suggested_correction="false"` is set for the reviewer to see
- If they disagree but the critic's `revised_confidence=0.5` → disagreement is flagged in `suggested_correction` but the actor's answer is kept

---

## 7. Response shapes

### Standard `/answers`

```json
{
  "answers": [
    {
      "question_id": "patient_bmi",
      "answer": "35.2 kg/m²",
      "explanation": "BMI of 35.2 kg/m² is stated in the visit notes.",
      "confidence": 0.95
    },
    {
      "question_id": "continuation",
      "answer": "true",
      "explanation": "Visit notes describe an ongoing medication regimen.",
      "confidence": 0.85
    },
    {
      "question_id": "cont_duration",
      "answer": "ongoing",
      "explanation": "Prescription duration field states 'ongoing'.",
      "confidence": 0.8
    }
  ]
}
```

### Actor-critic `/answers/actor-critic`

```json
{
  "answers": [
    {
      "question_id": "continuation",
      "actor_answer": "true",
      "answer": "true",
      "explanation": "Visit notes describe ongoing therapy.",
      "actor_confidence": 0.85,
      "critic_confidence": 0.9,
      "critique": "Prescription notes explicitly state 'ongoing' duration.",
      "suggested_correction": null
    },
    {
      "question_id": "cont_duration",
      "actor_answer": "6 weeks",
      "answer": "ongoing",
      "explanation": "Prescription duration field states 'ongoing'.",
      "actor_confidence": 0.6,
      "critic_confidence": 0.88,
      "critique": "Duration field in prescription reads 'ongoing', not 6 weeks.",
      "suggested_correction": "ongoing"
    }
  ]
}
```

In the second answer above: actor said `"6 weeks"`, critic said `"ongoing"` with 0.88 confidence → threshold exceeded → `answer` is corrected to `"ongoing"`. The `actor_answer` field preserves the original for the reviewer.

---

## 8. Full call stack summary

```
Browser / curl
    │
    │  POST /answers  (JSON body)
    ▼
FastAPI validates body → AnswerInput
    │
    ▼
app/main.py  get_answers()
    ├── split into unconditional / conditional questions
    │
    ├── app/llm.py  answer_questions()        ← PASS 1 (unconditional)
    │       ├── patient.model_dump_json()
    │       ├── _format_questions()
    │       ├── _system_prompt()
    │       └── OpenAI beta.chat.completions.parse(response_format=LLMAnswerList)
    │               └── returns list[Answer]
    │
    ├── app/visibility.py  filter_visible_questions()
    │       └── _evaluate_visible_if() per conditional question
    │               └── _parse_condition() per clause
    │
    ├── app/llm.py  answer_questions()        ← PASS 2 (visible conditionals)
    │       └── (same as above)
    │
    └── return AnswerOutput(answers = pass1 + pass2)
            │
            ▼
        FastAPI serializes via Pydantic → JSON response
```

For `/answers/actor-critic`, `answer_questions()` is replaced by `actor_critic_answers()`, which internally calls both `answer_questions()` and `critique_answers()` before merging.

For `/answers/stream`, the route returns a `StreamingResponse` wrapping `_stream_with_visibility()`, which calls `stream_answers()` in a loop — one LLM call per question, each `yield` pushing an SSE event to the client.

---

## 9. Eval pipeline — labeled cases, ECE, and accuracy measurement

The eval pipeline (`scripts/eval_pipeline.py`) answers one question: **is the system actually producing correct answers, and are the confidence scores honest?**

It never modifies the model. It never feeds data back into any LLM. It is purely a measurement tool that runs after the fact.

---

### What labeled_cases.json contains

Each entry is a pre-written test case with a patient, a set of questions, and the correct answer for each question written in advance:

```json
{
  "description": "New patient — explicit vitals in notes, no prior therapy",
  "patient": { ... },
  "questions": [
    { "type": "text",    "key": "bmi",         "content": "What is the patient's BMI?" },
    { "type": "boolean", "key": "continuation", "content": "Is this a continuation of therapy?" }
  ],
  "ground_truth": [
    { "question_id": "bmi",          "answer": "37.4 kg/m²", "answerable": true },
    { "question_id": "continuation", "answer": "false",       "answerable": true }
  ]
}
```

The `ground_truth` answers are what a correct system *should* produce. They were written by hand — not by the model — based on what the patient's visit notes actually say. Some entries include a `note` field to document why an ambiguous case was labeled a particular way:

```json
{
  "question_id": "antiobesity_wt_mgmt_6m",
  "answer": "false",
  "answerable": true,
  "note": "Lifestyle work started Jan 2025, visit Aug 2025 — just over 6 months but the note does not use 'comprehensive program' language; conservative answer is false"
}
```

The model **never sees this file**. It only ever receives patient data and questions through the normal LLM calls. The labeled cases are compared against the model's output in Python after the fact.

---

### How the eval pipeline runs (scripts/eval_pipeline.py)

```
scripts/eval_pipeline.py
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Load all cases from labeled_cases.json         │
│  _load_cases()  — eval_pipeline.py : 160        │
│                                                 │
│  For each case:                                 │
│    patient + questions → actor_critic_answers() │
│    (same function used by the API endpoint)     │
└───────────────┬─────────────────────────────────┘
                │
                │  ac_answers = [ActorCriticAnswer(...), ...]
                ▼
┌─────────────────────────────────────────────────┐
│  For each answer, compare against ground truth  │
│                                                 │
│  Boolean questions → exact string match         │
│    "true" == "true"  → correct                  │
│    "true" == "false" → wrong                    │
│                                                 │
│  Text questions → LLM judge                     │
│    "37.4 kg/m²" vs "37.4"                       │
│    → ask judge agent if they mean the same fact  │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  Record two separate result sets:               │
│                                                 │
│  actor_result  — uses ac_ans.actor_answer       │
│                  (what the actor said before    │
│                   any critic correction)        │
│                                                 │
│  ac_result     — uses ac_ans.answer             │
│                  (final answer, possibly        │
│                   corrected by critic)          │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  Compute accuracy and ECE for both              │
│  Print calibration table                        │
│  Save raw results to eval_results.json          │
└─────────────────────────────────────────────────┘
```

---

### The judge agent — why text answers need an LLM to compare

Boolean answers are exact: `"true"` either equals `"false"` or it doesn't. Text answers are not. The ground truth for BMI might be `"37.4 kg/m²"` but the model might return `"37.4"` or `"37.4 kilograms per square meter"`. A string comparison would mark both as wrong even though they're factually identical.

The judge agent is a separate LLM call that decides whether two answers convey the same fact:

```python
# eval_pipeline.py : 114
class JudgeVerdict(BaseModel):
    is_correct: bool
    reasoning: str

_judge_agent = Agent(
    model,
    result_type=JudgeVerdict,
    system_prompt=(
        "You are a medical accuracy judge. Given a question, the correct answer, "
        "and a predicted answer, decide if they convey the same factual meaning. "
        "Minor formatting differences (units, capitalisation, numeric precision) "
        "are OK. Respond with is_correct=true only if the core fact matches."
    ),
)
```

```python
# eval_pipeline.py : 139
async def _matches_ground_truth(question_content, predicted, correct, question_type):
    if predicted.strip().lower() == correct.strip().lower():
        return True          # exact match — no LLM needed

    if question_type == "boolean":
        return False         # booleans must match exactly, no fuzzy comparison

    # text questions only — ask the judge
    result = await agent.run(
        f"Question: {question_content}\n"
        f"Correct answer: {correct}\n"
        f"Predicted answer: {predicted}"
    )
    return result.data.is_correct
```

The judge never sees the patient data — it only compares two strings for a given question. It is not improving the system, just being used as a flexible comparator for text.

---

### What ECE is and why accuracy alone isn't enough

**Accuracy** just counts how many answers were right. If the model gets 80% correct, that tells you how often it's right but nothing about whether you can trust its confidence scores.

**ECE (Expected Calibration Error)** measures whether confidence scores are honest. The idea: if the model says confidence 0.9 on 10 questions, it should be right on roughly 9 of them. If it's right on 6, the model is overconfident and the ECE will be high. A well-calibrated model has a low ECE — its confidence scores actually predict how likely it is to be correct.

The pipeline groups all answers into 5 confidence buckets (0.0–0.2, 0.2–0.4, 0.4–0.6, 0.6–0.8, 0.8–1.0) and checks each bucket:

```python
# eval_pipeline.py : 59
def ece(self, confidence_key: str = "actor") -> float:
    bins = [[] for _ in range(5)]
    for r in self.records:
        conf = r.actor_confidence if confidence_key == "actor" else r.critic_confidence
        idx = min(int(conf * 5), 4)       # which bucket does this confidence fall in?
        bins[idx].append((conf, r.is_correct))

    n = sum(len(b) for b in bins)
    ece = 0.0
    for b in bins:
        avg_conf = sum(c for c, _ in b) / len(b)   # average confidence in this bucket
        avg_acc  = sum(ok for _, ok in b) / len(b)  # actual accuracy in this bucket
        ece += (len(b) / n) * abs(avg_conf - avg_acc)  # weighted gap
    return ece
```

A concrete example:

```
Bucket 0.8–1.0:  10 answers with avg confidence 0.9
                  model was correct on 9 of them
                  avg_accuracy = 0.9
                  gap = |0.9 - 0.9| = 0.0  ← perfectly calibrated

Bucket 0.8–1.0:  10 answers with avg confidence 0.9
                  model was correct on 6 of them
                  avg_accuracy = 0.6
                  gap = |0.9 - 0.6| = 0.3  ← overconfident
```

ECE is the weighted average of those gaps across all buckets. Lower is better. 0.0 would be perfect calibration.

The pipeline computes ECE separately for the actor's self-reported confidence and the critic's revised confidence, then compares them:

```
Actor-only   — accuracy: 78% | ECE: 0.18
Actor-Critic — accuracy: 81% | ECE (critic): 0.11

✓ Critic confidence is better calibrated (ECE ↓ 0.07)
```

This is the actual question the eval is trying to answer: does the critic's independent confidence score correlate better with real accuracy than the actor's self-reported one?

---

### The honest limitation

With only 3 cases in `labeled_cases.json`, the ECE numbers are statistically meaningless. Each calibration bucket ends up with 1–2 data points. A single wrong answer flips a bucket's accuracy by 50%. The pipeline prints an explicit warning when the sample size is below 30:

```python
# eval_pipeline.py : 193
_MIN_EVAL_SAMPLES = 30

if total_questions < _MIN_EVAL_SAMPLES:
    print(
        f"WARNING: {total_questions} labeled samples is below the recommended "
        f"minimum of {_MIN_EVAL_SAMPLES}. ECE values will have high variance."
    )
```

The framework is correct — the math is right, the pipeline works, the results are saved to `eval_results.json`. The dataset just needs to grow before the numbers mean anything. Ideally the ground truth would also be validated by a clinician rather than written synthetically.

---

## 10. App startup — env.py, Settings, and Logfire

Before any request is handled, the app needs its API keys and optional observability wired up. This is all handled in `app/env.py`.

### Settings (app/env.py:12)

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    logfire_key: str | None = None
    llm_model: str = "gpt-4o"

settings = Settings()
```

`pydantic_settings.BaseSettings` automatically reads from the `.env` file and maps values to typed fields. `settings` is a module-level singleton — it's created once when the module is first imported and reused everywhere. Any code that does `from app.env import settings` gets the same instance. `llm_model` defaults to `"gpt-4o"` but can be overridden by setting `LLM_MODEL` in `.env` without touching the code.

### Logfire observability (app/env.py:25)

```python
def setup_observability(app: FastAPI) -> None:
    if settings.logfire_key:
        logfire.configure(token=settings.logfire_key)
        logfire.instrument_fastapi(app)
    else:
        print("Logfire key not found, observability will not be setup")
```

Called once in `main.py` immediately after the `FastAPI` app is created. If `LOGFIRE_TOKEN` is in `.env`, Logfire instruments the app — every request, response, and LLM call gets traced and sent to the Logfire dashboard. If the key is missing the app starts normally with no observability. The `if/else` means the key is truly optional and won't crash startup.

---

## 11. OpenAI client singleton (app/llm.py:22)

```python
_client: AsyncOpenAI | None = None

def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client
```

Every LLM call in `llm.py` goes through `get_client()`. The first call creates a single `AsyncOpenAI` instance and stores it in the module-level `_client`. Every subsequent call returns that same instance.

The reason this matters: `AsyncOpenAI` manages an HTTP connection pool internally. If you created a new client on every request, you'd open and close a new connection pool for every LLM call, which is expensive. The singleton pattern means all requests share one pool. FastAPI's async model means many requests can use this one client concurrently without blocking each other.

---

## 12. Patient endpoints (app/main.py:47–71)

Three routes serve patient data to the frontend. All three call the same private function to load the data:

```python
# app/main.py : 153
def _load_sample_patients() -> list[dict[str, Any]]:
    with open(_SAMPLE_DATA_PATH) as f:
        return json.load(f)
```

`_SAMPLE_DATA_PATH` points to `sample_data/patient_data.json`. This reads the file from disk on every call — there is no caching. For a test frontend with a small file this is fine, but in production you'd load it once at startup or serve from a real database.

### GET /patients — list all patients

```python
@app.get("/patients")
async def list_patients() -> list[PatientSummary]:
    patients = _load_sample_patients()
    return [
        PatientSummary(
            index=i,
            name=f"{p['first_name']} {p['last_name']}",
            medication=p["prescription"]["medication"],
        )
        for i, p in enumerate(patients)
    ]
```

Returns a trimmed `PatientSummary` for each patient — just index, full name, and medication. The frontend uses this to populate the patient dropdown. The full patient record is not returned here, only what's needed to display the list.

### GET /patients/random — pick one at random

```python
@app.get("/patients/random")
async def get_random_patient() -> Patient:
    patients = _load_sample_patients()
    return Patient(**random.choice(patients))
```

Loads all patients and picks one at random. Returns the full `Patient` model. `random.choice` is not seeded so the result genuinely varies each call.

### GET /patients/{index} — fetch by position

```python
@app.get("/patients/{index}")
async def get_patient(index: int) -> Patient:
    patients = _load_sample_patients()
    if index < 0 or index >= len(patients):
        raise HTTPException(status_code=404, detail="Patient index out of range")
    return Patient(**patients[index])
```

`{index}` in the path is captured by FastAPI and cast to `int` automatically. Bounds checking is explicit — negative indices and out-of-range indices both return 404. `Patient(**patients[index])` unpacks the raw dict into the Pydantic model, which validates all fields.

---

## 13. Static file serving and the health check (app/main.py:38–44, 129)

### Health check

```python
@app.get("/")
@app.get("/health")
async def root() -> dict[str, str]:
    return {
        "message": "Pharmacy Prior Authorization API is running",
        "status": "healthy",
    }
```

Both `/` and `/health` are mapped to the same function using two decorators stacked on top of each other. This is standard FastAPI — you can attach multiple routes to one handler. Used by the test suite to verify the app started correctly.

### Static files

```python
# app/main.py : 129
app.mount("/static", StaticFiles(directory="app/static"), name="static")
```

`app.mount` hands off all requests starting with `/static` to FastAPI's built-in file server, which serves everything in `app/static/`. The frontend HTML file is accessible at `http://localhost:8000/static/index.html`. Because the frontend is served by the same FastAPI process on the same port, all API calls from the HTML page go to the same origin — this is why CORS middleware is not needed.

---

## 14. Test setup — conftest.py

The tests never start a real server or make real network requests. `conftest.py` wires `httpx` directly to the FastAPI app in memory:

```python
# tests/conftest.py
@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
```

`ASGITransport(app=app)` tells `httpx` to call the FastAPI app's ASGI interface directly instead of opening a real HTTP connection. `base_url="http://test"` is a fake URL — it never resolves. The result is a fully functional HTTP client that behaves exactly like calling the real API, but entirely in-process with no network, no port, and no running server.

This is why the tests can run without an OpenAI API key — the `answer_questions` and `actor_critic_answers` functions are mocked via `unittest.mock.patch` before the request is made, so the LLM call is never reached.

```python
# tests/test_answers.py : 98
with patch("app.main.answer_questions", new_callable=AsyncMock) as mock_llm:
    mock_llm.side_effect = lambda patient, question_set: _fake_answers(
        question_set.questions
    )
    response = await client.post("/answers", json=payload.model_dump())
```

The patch target is `app.main.answer_questions` — not `app.llm.answer_questions`. This is important: Python's `patch` replaces the name as it exists in the module being tested. Since `main.py` does `from app.llm import answer_questions`, the name `answer_questions` inside `app.main` is a separate reference. Patching `app.llm.answer_questions` would replace the original but leave the copy in `app.main` unchanged, so the mock would never intercept the call.
