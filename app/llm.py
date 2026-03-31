import json
import logging
from collections.abc import AsyncGenerator
from datetime import date

from fastapi import HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.env import settings
from app.models import (
    ActorCriticAnswer,
    Answer,
    CritiqueResult,
    Patient,
    Question,
    QuestionSet,
)

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


class LLMAnswerList(BaseModel):
    answers: list[Answer]


def _system_prompt() -> str:
    today = date.today().isoformat()
    return f"""Today's date is {today}. You are a medical prior authorization assistant. Given patient data and a list of form questions, answer each question accurately based ONLY on the provided patient information.

The prior authorization is being requested for the specific medication named in the patient's prescription field. When any question references "the requested medication", "the current medication", or "the medication", interpret it as referring exclusively to that prescription medication — not to any prior medications, trials, or medication history mentioned in the visit notes.

For each question provide:
- "question_id": the exact key from the input question
- "answer": a concise answer appropriate for the question type. For boolean questions use "true" or "false". For text questions provide a direct factual answer.
- "explanation": one sentence citing which part of the patient data supports this answer
- "confidence": a float from 0.0 to 1.0 indicating how strongly the patient data supports your answer (1.0 = explicitly stated, 0.5 = inferred, 0.0 = no relevant data found)

If the patient data does not contain enough information to answer a question, set answer to "Unable to determine", explain what data is missing, and set confidence to 0.0."""


def _single_question_system_prompt() -> str:
    today = date.today().isoformat()
    return f"""Today's date is {today}. You are a medical prior authorization assistant. Given patient data and a single form question, answer the question accurately based ONLY on the provided patient information.

The prior authorization is being requested for the specific medication named in the patient's prescription field. When the question references "the requested medication", "the current medication", or "the medication", interpret it as referring exclusively to that prescription medication — not to any prior medications, trials, or medication history mentioned in the visit notes.

For the question provide:
- "question_id": the exact key from the input question
- "answer": a concise answer appropriate for the question type. For boolean questions use "true" or "false". For text questions provide a direct factual answer.
- "explanation": one sentence citing which part of the patient data supports this answer
- "confidence": a float from 0.0 to 1.0 (1.0 = explicitly stated, 0.5 = inferred, 0.0 = no relevant data)

If the patient data does not contain enough information, set answer to "Unable to determine", explain what data is missing, and set confidence to 0.0."""


def _format_questions(questions: list[Question]) -> str:
    return json.dumps(
        [
            {"question_id": q.key, "content": q.content, "type": q.type}
            for q in questions
        ],
        indent=2,
    )


async def answer_questions(patient: Patient, question_set: QuestionSet) -> list[Answer]:
    """Call LLM once with all questions and return structured answers."""
    if not question_set.questions:
        return []

    patient_json = patient.model_dump_json(indent=2)
    questions_json = _format_questions(question_set.questions)

    try:
        response = await get_client().beta.chat.completions.parse(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {
                    "role": "user",
                    "content": f"Patient Data:\n{patient_json}\n\nQuestions:\n{questions_json}",
                },
            ],
            response_format=LLMAnswerList,
        )
        result = response.choices[0].message.parsed
        return result.answers if result else []

    except Exception as e:
        logger.error("LLM batch call failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate answers") from e


_CORRECTION_CONFIDENCE_THRESHOLD = 0.7


class _CritiqueList(BaseModel):
    critiques: list[CritiqueResult]


def _critic_system_prompt() -> str:
    today = date.today().isoformat()
    return f"""Today's date is {today}. You are an independent medical prior authorization auditor. You have NOT seen any AI-generated answers. Answer each question yourself using ONLY the provided patient data.

The prior authorization is being requested for the specific medication named in the patient's prescription field. When any question references "the requested medication", "the current medication", or "the medication", interpret it as referring exclusively to that prescription medication — not to any prior medications, trials, or medication history mentioned in the visit notes.

For each question, return:
- "question_id": exact key from the input
- "independent_answer": your own answer based solely on the patient data. For boolean questions use "true" or "false". For text questions provide a direct factual answer. Use "Unable to determine" if the data is insufficient.
- "is_supported": true if patient data directly supports your answer, false if you are inferring or guessing
- "critique": one sentence citing the specific data that supports or contradicts answerability
- "revised_confidence": your calibrated confidence (0.0–1.0) that your answer is correct (1.0 = explicitly stated, 0.5 = reasonable inference, 0.0 = no relevant data)

Be conservative: ambiguous or missing data should lower confidence, not raise it."""


async def critique_answers(
    patient: Patient,
    questions: list[Question],
) -> list[CritiqueResult]:
    if not questions:
        return []

    patient_json = patient.model_dump_json(indent=2)
    questions_json = _format_questions(questions)

    try:
        response = await get_client().beta.chat.completions.parse(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _critic_system_prompt()},
                {
                    "role": "user",
                    "content": f"Patient Data:\n{patient_json}\n\nQuestions:\n{questions_json}",
                },
            ],
            response_format=_CritiqueList,
        )
        result = response.choices[0].message.parsed
        return result.critiques if result else []

    except Exception as e:
        logger.error("Critic LLM call failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to critique answers") from e


async def actor_critic_answers(
    patient: Patient,
    question_set: QuestionSet,
) -> list[ActorCriticAnswer]:
    actor_answers = await answer_questions(patient, question_set)
    critiques = await critique_answers(patient, question_set.questions)
    critique_map = {c.question_id: c for c in critiques}

    merged: list[ActorCriticAnswer] = []
    for ans in actor_answers:
        c = critique_map.get(ans.question_id)
        correction = None
        final_answer = ans.answer
        if c and c.independent_answer.lower() != ans.answer.lower():
            correction = c.independent_answer
            if c.revised_confidence >= _CORRECTION_CONFIDENCE_THRESHOLD:
                final_answer = c.independent_answer
        merged.append(
            ActorCriticAnswer(
                question_id=ans.question_id,
                actor_answer=ans.answer,
                answer=final_answer,
                explanation=ans.explanation,
                actor_confidence=ans.confidence,
                critic_confidence=c.revised_confidence if c else None,
                critique=c.critique if c else None,
                suggested_correction=correction,
            )
        )
    return merged


async def stream_answers(
    patient: Patient, questions: list[Question]
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted answers one question at a time."""
    patient_json = patient.model_dump_json(indent=2)

    for question in questions:
        question_json = json.dumps(
            {
                "question_id": question.key,
                "content": question.content,
                "type": question.type,
            },
            indent=2,
        )
        try:
            response = await get_client().beta.chat.completions.parse(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": _single_question_system_prompt()},
                    {
                        "role": "user",
                        "content": f"Patient Data:\n{patient_json}\n\nQuestion:\n{question_json}",
                    },
                ],
                response_format=Answer,
            )
            answer = response.choices[0].message.parsed
            if answer is None:
                answer = Answer(
                    question_id=question.key,
                    answer="Unable to determine",
                    confidence=0.0,
                )
            yield f"data: {answer.model_dump_json()}\n\n"

        except Exception as e:
            logger.error("LLM stream call failed for question %s: %s", question.key, e)
            error_answer = Answer(
                question_id=question.key,
                answer="Unable to determine",
                explanation=f"LLM error: {e}",
                confidence=0.0,
            )
            yield f"data: {error_answer.model_dump_json()}\n\n"
