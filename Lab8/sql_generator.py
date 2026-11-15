import os
import re

from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError

load_dotenv()

MODEL = "gpt-4o-mini"


def build_system_prompt(dialect: str, schema: str) -> str:
    return (
        f"You are a strict SQL query generator. Generate a valid query in {dialect} dialect.\n"
        f"Schema Context:\n{schema}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Output ONLY the clean SQL query.\n"
        "2. Absolute zero greetings, zero intro/outro text, zero markdown block wraps (no ```sql ... ```).\n"
        "3. Return raw database query only."
    )


def clean_sql(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def generate_sql(api_key: str, dialect: str, schema: str, user_query: str) -> str:
    effective_key = api_key.strip() if api_key and api_key.strip() else os.getenv("OPENAI_API_KEY")
    if not effective_key:
        return "ПОМИЛКА: Вкажіть API Key у полі або у файлі .env"
    if not schema or not schema.strip():
        return "ПОМИЛКА: Схема бази даних порожня."
    if not user_query or not user_query.strip():
        return "ПОМИЛКА: Текстовий запит порожній."

    try:
        client = OpenAI(api_key=effective_key)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt(dialect, schema)},
                {"role": "user", "content": user_query},
            ],
            temperature=0.0,
        )
        return clean_sql(response.choices[0].message.content or "")

    except AuthenticationError:
        return "ПОМИЛКА: Невірний API Key."
    except Exception as e:
        return f"ПОМИЛКА API: {e}"