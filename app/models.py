from typing import Literal

from pydantic import BaseModel


class Question(BaseModel):
    type: Literal["text", "boolean"]
    key: str
    content: str
    visible_if: str | None = None


class QuestionSet(BaseModel):
    name: str
    questions: list[Question]


class Answer(BaseModel):
    question_id: str
    answer: str
    explanation: str | None = None
    confidence: float | None = None


class Prescription(BaseModel):
    medication: str
    dosage: str
    frequency: str
    duration: str


class Patient(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: str
    gender: str
    prescription: Prescription
    visit_notes: list[str]


class AnswerInput(BaseModel):
    patient: Patient
    question_set: QuestionSet


class AnswerOutput(BaseModel):
    answers: list[Answer]


class PatientSummary(BaseModel):
    index: int
    name: str
    medication: str


class CritiqueResult(BaseModel):
    question_id: str
    independent_answer: str
    is_supported: bool
    critique: str
    revised_confidence: float


class ActorCriticAnswer(BaseModel):
    question_id: str
    actor_answer: str
    answer: str
    explanation: str | None = None
    actor_confidence: float | None = None
    critic_confidence: float | None = None
    critique: str | None = None
    suggested_correction: str | None = None


class ActorCriticOutput(BaseModel):
    answers: list[ActorCriticAnswer]
