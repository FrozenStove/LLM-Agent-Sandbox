from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.env import settings  # noqa: E402
from app.llm import actor_critic_answers  # noqa: E402
from app.models import Patient, Prescription, Question, QuestionSet  # noqa: E402

_LABELED_CASES_PATH = (
    Path(__file__).parent.parent / "sample_data" / "labeled_cases.json"
)


@dataclass
class GroundTruth:
    question_id: str
    answer: str
    answerable: bool
    note: str = ""


@dataclass
class EvalCase:
    description: str
    patient: Patient
    question_set: QuestionSet
    ground_truth: list[GroundTruth]


@dataclass
class AnswerRecord:
    question_id: str
    predicted_answer: str
    correct_answer: str
    is_correct: bool
    actor_confidence: float | None
    critic_confidence: float | None


@dataclass
class PassResult:
    records: list[AnswerRecord] = field(default_factory=list)

    def accuracy(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.is_correct for r in self.records) / len(self.records)

    def ece(self, confidence_key: str = "actor") -> float:
        bins: list[list[tuple[float, bool]]] = [[] for _ in range(5)]
        for r in self.records:
            conf = (
                r.actor_confidence if confidence_key == "actor" else r.critic_confidence
            )
            if conf is None:
                continue
            idx = min(int(conf * 5), 4)
            bins[idx].append((conf, r.is_correct))

        n = sum(len(b) for b in bins)
        if n == 0:
            return 0.0
        ece = 0.0
        for b in bins:
            if not b:
                continue
            avg_conf = sum(c for c, _ in b) / len(b)
            avg_acc = sum(correct for _, correct in b) / len(b)
            ece += (len(b) / n) * abs(avg_conf - avg_acc)
        return ece

    def calibration_table(
        self, confidence_key: str = "actor"
    ) -> list[dict[str, str | int]]:
        labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
        bins: list[list[tuple[float, bool]]] = [[] for _ in range(5)]
        for r in self.records:
            conf = (
                r.actor_confidence if confidence_key == "actor" else r.critic_confidence
            )
            if conf is None:
                continue
            idx = min(int(conf * 5), 4)
            bins[idx].append((conf, r.is_correct))

        rows: list[dict[str, str | int]] = []
        for label, b in zip(labels, bins, strict=False):
            if not b:
                rows.append(
                    {"bucket": label, "n": 0, "avg_confidence": "-", "accuracy": "-"}
                )
                continue
            rows.append(
                {
                    "bucket": label,
                    "n": len(b),
                    "avg_confidence": f"{sum(c for c, _ in b) / len(b):.2f}",
                    "accuracy": f"{sum(ok for _, ok in b) / len(b):.2f}",
                }
            )
        return rows


class JudgeVerdict(BaseModel):
    is_correct: bool
    reasoning: str


_judge_agent: Agent[None, JudgeVerdict] | None = None


def _get_judge_agent() -> Agent[None, JudgeVerdict]:
    global _judge_agent
    if _judge_agent is None:
        model = OpenAIModel(settings.llm_model, api_key=settings.openai_api_key)
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
    return _judge_agent


async def _matches_ground_truth(
    question_content: str,
    predicted: str,
    correct: str,
    question_type: str,
) -> bool:
    predicted_norm = predicted.strip().lower()
    correct_norm = correct.strip().lower()
    if predicted_norm == correct_norm:
        return True
    if question_type == "boolean":
        return False
    agent = _get_judge_agent()
    result = await agent.run(
        f"Question: {question_content}\n"
        f"Correct answer: {correct}\n"
        f"Predicted answer: {predicted}"
    )
    return result.data.is_correct


def _load_cases() -> list[EvalCase]:
    raw = json.loads(_LABELED_CASES_PATH.read_text())
    cases: list[EvalCase] = []
    for item in raw:
        patient = Patient(
            first_name=item["patient"]["first_name"],
            last_name=item["patient"]["last_name"],
            date_of_birth=item["patient"]["date_of_birth"],
            gender=item["patient"]["gender"],
            prescription=Prescription(**item["patient"]["prescription"]),
            visit_notes=item["patient"]["visit_notes"],
        )
        questions = [Question(**q) for q in item["questions"]]
        ground_truth = [
            GroundTruth(
                question_id=gt["question_id"],
                answer=gt["answer"],
                answerable=gt["answerable"],
                note=gt.get("note", ""),
            )
            for gt in item["ground_truth"]
        ]
        cases.append(
            EvalCase(
                description=item["description"],
                patient=patient,
                question_set=QuestionSet(name=item["description"], questions=questions),
                ground_truth=ground_truth,
            )
        )
    return cases


_MIN_EVAL_SAMPLES = 30


async def run_eval() -> None:
    cases = _load_cases()
    total_questions = sum(len(c.ground_truth) for c in cases)
    print("\n=== Actor-Critic Calibration Evaluation ===")
    print(f"Cases: {len(cases)} | Total questions: {total_questions}\n")
    if total_questions < _MIN_EVAL_SAMPLES:
        print(
            f"WARNING: {total_questions} labeled samples is below the recommended "
            f"minimum of {_MIN_EVAL_SAMPLES}. ECE values will have high variance.\n"
        )

    actor_result = PassResult()
    ac_result = PassResult()

    for case in cases:
        print(f"── Case: {case.description}")
        gt_map = {gt.question_id: gt for gt in case.ground_truth}
        q_map = {q.key: q for q in case.question_set.questions}

        ac_answers = await actor_critic_answers(case.patient, case.question_set)

        for ac_ans in ac_answers:
            gt = gt_map.get(ac_ans.question_id)
            q = q_map.get(ac_ans.question_id)
            if not gt or not q:
                continue

            actor_correct = await _matches_ground_truth(
                q.content, ac_ans.actor_answer, gt.answer, q.type
            )
            actor_result.records.append(
                AnswerRecord(
                    question_id=ac_ans.question_id,
                    predicted_answer=ac_ans.actor_answer,
                    correct_answer=gt.answer,
                    is_correct=actor_correct,
                    actor_confidence=ac_ans.actor_confidence,
                    critic_confidence=None,
                )
            )

            ac_correct = await _matches_ground_truth(
                q.content, ac_ans.answer, gt.answer, q.type
            )
            ac_result.records.append(
                AnswerRecord(
                    question_id=ac_ans.question_id,
                    predicted_answer=ac_ans.answer,
                    correct_answer=gt.answer,
                    is_correct=ac_correct,
                    actor_confidence=ac_ans.actor_confidence,
                    critic_confidence=ac_ans.critic_confidence,
                )
            )

            actor_flag = "✓" if actor_correct else "✗"
            ac_flag = "✓" if ac_correct else "✗"
            print(
                f"   {ac_ans.question_id:<35}"
                f"  actor={actor_flag} [{ac_ans.actor_confidence or 0:.2f}]"
                f"  ac={ac_flag} [{ac_ans.critic_confidence or 0:.2f}]"
                f"  gt={gt.answer!r}"
            )
        print()

    _print_calibration_analysis(actor_result, ac_result)


def _print_calibration_analysis(
    actor_result: PassResult, ac_result: PassResult
) -> None:
    print("=== Calibration Analysis ===\n")

    def _print_table(title: str, rows: list[dict[str, str | int]]) -> None:
        print(f"  {title}")
        print(f"  {'Bucket':<10} {'N':>4} {'Avg Conf':>10} {'Accuracy':>10}")
        print(f"  {'-' * 38}")
        for r in rows:
            print(
                f"  {r['bucket']:<10} {r['n']:>4} {r['avg_confidence']:>10} {r['accuracy']:>10}"
            )
        print()

    _print_table(
        "Actor self-reported confidence vs accuracy:",
        actor_result.calibration_table("actor"),
    )
    _print_table(
        "Critic-assigned confidence vs accuracy:",
        ac_result.calibration_table("critic"),
    )

    actor_ece = actor_result.ece("actor")
    critic_ece = ac_result.ece("critic")
    actor_acc = actor_result.accuracy()
    ac_acc = ac_result.accuracy()

    print(f"  Actor-only  — accuracy: {actor_acc:.0%} | ECE: {actor_ece:.3f}")
    print(f"  Actor-Critic — accuracy: {ac_acc:.0%} | ECE (critic): {critic_ece:.3f}")
    delta_ece = actor_ece - critic_ece
    print()
    if delta_ece > 0:
        print(f"  ✓ Critic confidence is better calibrated (ECE ↓ {delta_ece:.3f})")
    elif delta_ece < 0:
        print(
            f"  ✗ Self-reported confidence is better calibrated (ECE ↑ {abs(delta_ece):.3f})"
        )
    else:
        print("  = Calibration is equivalent between actor and critic.")

    print()
    print("Note: with a small labeled set the ECE buckets may be sparse.")
    print("Scale the labeled_cases.json dataset for statistically meaningful results.")

    out_path = Path(__file__).parent.parent / "eval_results.json"
    out_path.write_text(
        json.dumps(
            {
                "actor_records": [
                    {
                        "question_id": r.question_id,
                        "is_correct": r.is_correct,
                        "actor_confidence": r.actor_confidence,
                    }
                    for r in actor_result.records
                ],
                "ac_records": [
                    {
                        "question_id": r.question_id,
                        "is_correct": r.is_correct,
                        "actor_confidence": r.actor_confidence,
                        "critic_confidence": r.critic_confidence,
                    }
                    for r in ac_result.records
                ],
                "summary": {
                    "actor_accuracy": actor_acc,
                    "ac_accuracy": ac_acc,
                    "actor_ece": actor_ece,
                    "critic_ece": critic_ece,
                },
            },
            indent=2,
        )
    )
    print(f"\nRaw results saved to {out_path.name} for plotting.")


if __name__ == "__main__":
    asyncio.run(run_eval())
