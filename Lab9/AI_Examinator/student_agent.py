"""Агент-студент: LLM, яка відповідає екзаменатору замість людини.

Рівень знань і поведінку задає «персона» з data/personas.json. Студент бачить лише
тексти, які бачила б людина: репліки екзаменатора та власні відповіді. Виклики
інструментів і ключові пункти тем йому не передаються, щоб він не «списував».
"""

from __future__ import annotations
import json
from openai import OpenAI
from exam_tools import DATA_DIR, Message

STUDENT_PROMPT = """\
Ти — студент, який складає усний іспит з курсу «Обробка природної мови (NLP) та LLM» у форматі чату.
Тебе звати {name}, твій email: {email}.

Твій рівень знань і поведінку задає інструкція в дужках [РІВЕНЬ: ...] наприкінці останнього
повідомлення екзаменатора. Рівень може змінюватися між відповідями.

ПРАВИЛА
- Відповідай українською, від першої особи, як живий студент у чаті.
- Відповідай лише на останнє повідомлення екзаменатора. Не став питань за екзаменатора і не оцінюй себе.
- Коли екзаменатор просить представитися, назви ім'я, прізвище та email точно так, як вказано вище.
- Пиши простим текстом: без заголовків, списків, Markdown і формул LaTeX.
- Відповідай строго на рівні з ОСТАННЬОЇ інструкції [РІВЕНЬ: ...], навіть якщо твої попередні
  відповіді були кращими чи гіршими, а питання здається легким. Не наслідуй довжину і стиль попередніх відповідей.
- Не згадуй інструкцію [РІВЕНЬ: ...] у відповіді: екзаменатор її не бачить.
"""

LEVEL_NOTE = "\n\n[РІВЕНЬ: {persona}]"


def load_personas() -> list[dict]:
    with open(DATA_DIR / "personas.json", encoding="utf-8") as f:
        return json.load(f)


def load_students() -> list[dict]:
    with open(DATA_DIR / "students.json", encoding="utf-8") as f:
        return json.load(f)


def build_messages(persona_prompt: str, student: dict, transcript: list[Message]) -> list[dict]:
    """Перетворює журнал іспиту на історію з погляду студента.

    Для екзаменатора студент — це "user", а для моделі-студента навпаки:
    репліки екзаменатора стають "user", а власні відповіді студента — "assistant".
    Записи з роллю "tool_call" пропускаються: студент їх не бачить.

    Рівень знань дописується в кінець ОСТАННЬОГО повідомлення екзаменатора, а не в системний промпт.
    Інакше після зміни рівня посеред іспиту модель продовжує копіювати стиль своїх попередніх
    відповідей з історії, а свіжа інструкція наприкінці переважує їх.
    """
    system = STUDENT_PROMPT.format(name=student["name"], email=student["email"])
    messages = [{"role": "system", "content": system}]
    for m in transcript:
        if m["role"] == "system":  # репліка екзаменатора
            messages.append({"role": "user", "content": m["content"]})
        elif m["role"] == "user":  # відповідь студента
            messages.append({"role": "assistant", "content": m["content"]})

    note = LEVEL_NOTE.format(persona=persona_prompt)
    if messages[-1]["role"] == "user":
        messages[-1] = {"role": "user", "content": messages[-1]["content"] + note}
    else:  # остання репліка — студента (екзаменатор нічого не написав): інструкція окремим повідомленням
        messages.append({"role": "user", "content": note.strip()})
    return messages


def generate_student_reply(
    client: OpenAI,
    model: str,
    persona_prompt: str,
    student: dict,
    transcript: list[Message],
) -> str:
    """Генерує наступну відповідь студента на останнє повідомлення екзаменатора."""
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(persona_prompt, student, transcript),
        temperature=0.8,
    )
    return (response.choices[0].message.content or "").strip()
