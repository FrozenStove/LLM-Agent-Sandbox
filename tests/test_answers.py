from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.models import (
    ActorCriticAnswer,
    Answer,
    AnswerInput,
    Patient,
    Prescription,
    Question,
    QuestionSet,
)


def make_prescription(**overrides: object) -> Prescription:
    defaults: dict[str, object] = {
        "medication": "Zepbound",
        "dosage": "5 mg",
        "frequency": "once weekly",
        "duration": "ongoing",
    }
    return Prescription(**(defaults | overrides))


def make_patient(**overrides: object) -> Patient:
    defaults: dict[str, object] = {
        "first_name": "Jane",
        "last_name": "Doe",
        "date_of_birth": "1985-03-15",
        "gender": "Female",
        "prescription": make_prescription(),
        "visit_notes": ["Patient presents for weight management. BMI 35.2 kg/m²."],
    }
    return Patient(**(defaults | overrides))


def make_question(**overrides: object) -> Question:
    defaults: dict[str, object] = {
        "type": "text",
        "key": "patient_bmi",
        "content": "What is the patient's BMI?",
    }
    return Question(**(defaults | overrides))


def make_answer_input(**overrides: object) -> AnswerInput:
    defaults: dict[str, object] = {
        "patient": make_patient(),
        "question_set": QuestionSet(
            name="Test Question Set",
            questions=[make_question()],
        ),
    }
    return AnswerInput(**(defaults | overrides))


def _fake_answers(questions: list[Question]) -> list[Answer]:
    """Return a fake Answer for each question, used by the LLM mock."""
    return [
        Answer(
            question_id=q.key,
            answer="35.2 kg/m²" if q.type == "text" else "true",
            explanation="Extracted from visit notes.",
            confidence=0.9,
        )
        for q in questions
    ]


async def test_health_check(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Pharmacy Prior Authorization API is running",
        "status": "healthy",
    }


async def test_answers_returns_correct_structure(client: AsyncClient) -> None:
    payload = make_answer_input(
        question_set=QuestionSet(
            name="Test Set",
            questions=[
                make_question(key="patient_bmi", content="What is the patient's BMI?"),
                make_question(
                    key="tried_lifestyle",
                    type="boolean",
                    content="Has the patient tried lifestyle modifications?",
                ),
            ],
        )
    )

    # Patch the name as imported in app.main so calls inside the route are intercepted.
    with patch("app.main.answer_questions", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = lambda patient, question_set: _fake_answers(
            question_set.questions
        )
        response = await client.post("/answers", json=payload.model_dump())

    assert response.status_code == 200
    data = response.json()
    assert "answers" in data
    assert len(data["answers"]) == len(payload.question_set.questions)
    for answer in data["answers"]:
        assert "question_id" in answer
        assert "answer" in answer


async def test_empty_questions_returns_empty_answers(client: AsyncClient) -> None:
    payload = make_answer_input(
        question_set=QuestionSet(name="Empty Set", questions=[])
    )
    response = await client.post("/answers", json=payload.model_dump())
    assert response.status_code == 200
    assert response.json() == {"answers": []}


async def test_answers_include_explanation_and_confidence(client: AsyncClient) -> None:
    payload = make_answer_input()

    with patch("app.main.answer_questions", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = lambda patient, question_set: _fake_answers(
            question_set.questions
        )
        response = await client.post("/answers", json=payload.model_dump())

    assert response.status_code == 200
    for answer in response.json()["answers"]:
        assert "explanation" in answer
        assert "confidence" in answer


@pytest.fixture
async def sse_client(client: AsyncClient) -> AsyncGenerator[AsyncClient, None]:
    yield client


def _fake_actor_critic_answers(questions: list[Question]) -> list[ActorCriticAnswer]:
    return [
        ActorCriticAnswer(
            question_id=q.key,
            actor_answer="35.2 kg/m²" if q.type == "text" else "true",
            answer="35.2 kg/m²" if q.type == "text" else "true",
            explanation="Extracted from visit notes.",
            actor_confidence=0.8,
            critic_confidence=0.85,
            critique="Answer is directly supported by the BMI stated in visit notes.",
        )
        for q in questions
    ]


async def test_actor_critic_returns_correct_structure(client: AsyncClient) -> None:
    payload = make_answer_input(
        question_set=QuestionSet(
            name="Test Set",
            questions=[
                make_question(key="patient_bmi", content="What is the patient's BMI?"),
                make_question(
                    key="tried_lifestyle",
                    type="boolean",
                    content="Has the patient tried lifestyle modifications?",
                ),
            ],
        )
    )

    with patch("app.main.actor_critic_answers", new_callable=AsyncMock) as mock_ac:
        mock_ac.side_effect = lambda patient, question_set: _fake_actor_critic_answers(
            question_set.questions
        )
        response = await client.post("/answers/actor-critic", json=payload.model_dump())

    assert response.status_code == 200
    data = response.json()
    assert "answers" in data
    assert len(data["answers"]) == len(payload.question_set.questions)
    for answer in data["answers"]:
        assert "question_id" in answer
        assert "actor_answer" in answer
        assert "answer" in answer
        assert "actor_confidence" in answer
        assert "critic_confidence" in answer
        assert "critique" in answer


async def test_actor_critic_empty_questions(client: AsyncClient) -> None:
    payload = make_answer_input(
        question_set=QuestionSet(name="Empty Set", questions=[])
    )
    response = await client.post("/answers/actor-critic", json=payload.model_dump())
    assert response.status_code == 200
    assert response.json() == {"answers": []}


async def test_stream_endpoint_returns_sse_events(client: AsyncClient) -> None:
    payload = make_answer_input()

    fake_answer = Answer(
        question_id="patient_bmi",
        answer="35.2 kg/m²",
        explanation="Extracted from visit notes.",
        confidence=0.9,
    )

    async def fake_stream(patient, questions):
        for _q in questions:
            yield f"data: {fake_answer.model_dump_json()}\n\n"

    with patch("app.main.stream_answers", side_effect=fake_stream):
        response = await client.post("/answers/stream", json=payload.model_dump())

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
