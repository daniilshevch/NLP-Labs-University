"""Функції іспиту, які викликає агент.

Замість бази даних використовуються JSON-файли:
  data/students.json  — список студентів
  data/topics.json    — список тем з ключовими пунктами для оцінювання
  results/exam_log.jsonl — журнал початку/завершення іспитів
  results/<email>_<час>.json — повний результат кожного іспиту

Стан конкретного іспиту (обрані теми, поточна тема) зберігається в об'єкті
ExamSession, тому кілька студентів можуть складати іспит одночасно.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

TOPICS_PER_EXAM = 3


class Message(TypedDict):
    role: Literal["system", "user", "tool_call"]
    content: str
    datetime: str


class ExamError(Exception):
    """Помилка, текст якої повертається моделі як результат інструмента."""


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_json(name: str):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _log_event(event: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exam_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def find_student(email: str, name: str) -> dict:
    students = _load_json("students.json")
    students_by_email = {_normalize(s["email"]): s for s in students}
    student = students_by_email.get(_normalize(email))
    if student is None:
        raise ExamError(
            f"Студента з email '{email}' не знайдено в базі. "
            "Попроси студента перевірити email."
        )

    if sorted(_normalize(name).split()) != sorted(_normalize(student["name"]).split()):
        raise ExamError(
            "Email знайдено, але ім'я не збігається з базою. "
            "Попроси студента уточнити ім'я та прізвище."
        )
    return student


def topic_key_points(title: str) -> list[str]:
    for topic in _load_json("topics.json"):
        if topic["title"] == title:
            return topic.get("key_points", [])
    return []


@dataclass
class ExamSession:
    """Стан одного іспиту. Функції нижче мають сигнатури із завдання."""

    email: str | None = None
    name: str | None = None
    topics: list[str] = field(default_factory=list)
    next_index: int = 0
    current_topic: str | None = None
    topic_results: dict[str, dict] = field(default_factory=dict)
    started_at: str | None = None
    finished: bool = False


    def start_exam(self, email: str, name: str) -> list[str]:
        """Перевіряє студента, записує початок іспиту і повертає 2-3 теми."""
        if self.finished:
            raise ExamError("Іспит уже завершено.")
        if self.email is not None:
            raise ExamError(f"Іспит уже розпочато для {self.email}. Теми: {self.topics}")

        student = find_student(email, name)
        all_topics = [t["title"] for t in _load_json("topics.json")]

        self.email = student["email"]
        self.name = student["name"]
        self.topics = random.sample(all_topics, k=min(TOPICS_PER_EXAM, len(all_topics)))
        self.started_at = now_iso()

        _log_event({
            "event": "start_exam",
            "email": self.email,
            "name": self.name,
            "topics": self.topics,
            "datetime": self.started_at,
        })
        return self.topics

    def get_next_topic(self) -> str:
        """Повертає назву наступної теми або порожній рядок, якщо теми скінчилися."""
        if self.email is None:
            raise ExamError("Спочатку потрібно викликати start_exam.")
        if self.current_topic and self.current_topic not in self.topic_results:
            raise ExamError(
                f"Тему '{self.current_topic}' ще не оцінено. "
                "Спочатку виклич record_topic_result для неї."
            )
        if self.next_index >= len(self.topics):
            self.current_topic = None
            return ""
        self.current_topic = self.topics[self.next_index]
        self.next_index += 1
        return self.current_topic

    def end_exam(
        self,
        email: str,
        score: float,
        history: list[Message],
        feedback: str = "",
    ) -> None:
        """Записує результат іспиту у файл."""
        if self.email is None:
            raise ExamError("Іспит ще не розпочато.")
        if self.finished:
            raise ExamError("Іспит уже завершено, результат записано.")
        if _normalize(email) != _normalize(self.email):
            raise ExamError(f"Email не збігається з email іспиту ({self.email}).")
        score = float(score)
        if not 0 <= score <= 10:
            raise ExamError("Оцінка має бути в межах від 0 до 10.")

        finished_at = now_iso()
        result = {
            "email": self.email,
            "name": self.name,
            "score": round(score, 1),
            "feedback": feedback,
            "topics": self.topics,
            "topic_results": self.topic_results,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "history": history,
        }
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = finished_at.replace(":", "-")
        path = RESULTS_DIR / f"{self.email}_{stamp}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        _log_event({
            "event": "end_exam",
            "email": self.email,
            "score": result["score"],
            "datetime": finished_at,
        })
        print(f"[end_exam] {self.email}: {result['score']}/10 -> {path}")
        self.finished = True


    def record_topic_result(self, topic: str, score: float, comment: str) -> str:
        """Фіксує оцінку за одну тему, щоб підсумкова оцінка була обґрунтованою."""
        if topic not in self.topics:
            raise ExamError(f"Теми '{topic}' немає серед тем іспиту: {self.topics}")
        score = float(score)
        if not 0 <= score <= 10:
            raise ExamError("Оцінка за тему має бути в межах від 0 до 10.")
        self.topic_results[topic] = {"score": score, "comment": comment}
        return f"Оцінку за тему '{topic}' записано: {score}/10."

    @property
    def remaining_topics(self) -> int:
        return len(self.topics) - self.next_index