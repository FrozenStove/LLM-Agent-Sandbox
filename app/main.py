import json
import logging
import random
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.env import setup_observability
from app.llm import actor_critic_answers, answer_questions, stream_answers
from app.models import (
    ActorCriticOutput,
    AnswerInput,
    AnswerOutput,
    Patient,
    PatientSummary,
    Question,
    QuestionSet,
)
from app.visibility import filter_visible_questions

logger = logging.getLogger(__name__)

_SAMPLE_DATA_PATH = Path(__file__).parent.parent / "sample_data" / "patient_data.json"

app = FastAPI(
    title="Pharmacy Prior Authorization API",
    description="API for generating answers to prior authorization questions using patient data",
    version="1.0.0",
)

setup_observability(app)


@app.get("/")
@app.get("/health")
async def root() -> dict[str, str]:
    return {
        "message": "Pharmacy Prior Authorization API is running",
        "status": "healthy",
    }


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


@app.get("/patients/random")
async def get_random_patient() -> Patient:
    patients = _load_sample_patients()
    return Patient(**random.choice(patients))


@app.get("/patients/{index}")
async def get_patient(index: int) -> Patient:
    patients = _load_sample_patients()
    if index < 0 or index >= len(patients):
        raise HTTPException(status_code=404, detail="Patient index out of range")
    return Patient(**patients[index])


@app.post("/answers")
async def get_answers(data: AnswerInput) -> AnswerOutput:
    all_questions = data.question_set.questions
    unconditional = [q for q in all_questions if q.visible_if is None]
    first_answers = await answer_questions(
        data.patient,
        data.question_set.__class__(
            name=data.question_set.name, questions=unconditional
        ),
    )
    current_answers = {a.question_id: a.answer for a in first_answers}
    conditional = [q for q in all_questions if q.visible_if is not None]
    visible_conditional = filter_visible_questions(conditional, current_answers)
    if not visible_conditional:
        return AnswerOutput(answers=first_answers)
    second_answers = await answer_questions(
        data.patient,
        data.question_set.__class__(
            name=data.question_set.name, questions=visible_conditional
        ),
    )
    return AnswerOutput(answers=first_answers + second_answers)


@app.post("/answers/actor-critic")
async def get_answers_actor_critic(data: AnswerInput) -> ActorCriticOutput:
    all_questions = data.question_set.questions
    unconditional = [q for q in all_questions if q.visible_if is None]
    first_answers = await actor_critic_answers(
        data.patient,
        QuestionSet(name=data.question_set.name, questions=unconditional),
    )
    current_answers = {a.question_id: a.answer for a in first_answers}
    conditional = [q for q in all_questions if q.visible_if is not None]
    visible_conditional = filter_visible_questions(conditional, current_answers)
    if not visible_conditional:
        return ActorCriticOutput(answers=first_answers)
    second_answers = await actor_critic_answers(
        data.patient,
        QuestionSet(name=data.question_set.name, questions=visible_conditional),
    )
    return ActorCriticOutput(answers=first_answers + second_answers)


@app.post("/answers/stream")
async def answers_stream(data: AnswerInput) -> StreamingResponse:
    all_questions = data.question_set.questions
    unconditional = [q for q in all_questions if q.visible_if is None]
    conditional = [q for q in all_questions if q.visible_if is not None]
    return StreamingResponse(
        _stream_with_visibility(data.patient, unconditional, conditional),
        media_type="text/event-stream",
    )


app.mount("/static", StaticFiles(directory="app/static"), name="static")


async def _stream_with_visibility(
    patient: Patient,
    unconditional: list[Question],
    conditional: list[Question],
) -> AsyncGenerator[str, None]:
    current_answers: dict[str, str] = {}
    async for event in stream_answers(patient, unconditional):
        if event.startswith("data: "):
            try:
                payload = json.loads(event[6:])
                current_answers[payload["question_id"]] = payload["answer"]
            except Exception:
                logger.warning(
                    "Failed to parse SSE event for visibility tracking: %s", event
                )
        yield event
    visible_conditional = filter_visible_questions(conditional, current_answers)
    async for event in stream_answers(patient, visible_conditional):
        yield event


def _load_sample_patients() -> list[dict[str, Any]]:
    with open(_SAMPLE_DATA_PATH) as f:
        return json.load(f)
