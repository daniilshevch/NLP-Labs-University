import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def generate_sql(api_key: str, dialect: str, schema: str, user_query: str) -> str:
    effective_key = api_key.strip() if api_key and api_key.strip() else os.getenv("OPENAI_API_KEY")
    if not effective_key:
        return "-- ПОМИЛКА: Вкажіть API Key у полі або у файлі .env"
    if not user_query or not user_query.strip():
        return "ПОМИЛКА: Текстовий запит порожній."

    try:
        client = OpenAI(api_key=effective_key)

        system_prompt = (
            f"You are a strict SQL query generator. Generate a valid query in {dialect} dialect.\n"
            f"Schema Context:\n{schema}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Output ONLY the clean SQL query.\n"
            "2. Absolute zero greetings, zero intro/outro text, zero markdown block wraps (no ```sql ... ```).\n"
            "3. Return raw database query only."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0
        )

        result = response.choices[0].message.content.strip()
        return result

    except Exception as e:
        return f"-- ПОМИЛКА API: {str(e)}"


DEFAULT_SCHEMA = """CREATE TABLE Users (
    Id INT PRIMARY KEY,
    FullName VARCHAR(100),
    Email VARCHAR(100),
    CreatedAt DATETIME
);

CREATE TABLE Orders (
    Id INT PRIMARY KEY,
    UserId INT FOREIGN KEY REFERENCES Users(Id),
    TotalAmount DECIMAL(10, 2),
    Status VARCHAR(20),
    OrderDate DATETIME
);"""

DEFAULT_QUERY = "Знайди суму всіх сплачених замовлень (Status = 'Paid') за останній місяць для кожного користувача"


with gr.Blocks(title="NL to SQL Generator") as demo:
    gr.Markdown("# Конвертатор текстових запитів у SQL")
    gr.Markdown(
        "Сервіс для автоматичної генерації SQL-запитів із текстових вимог природною мовою "
        "на основі наданої схеми БД за допомогою моделі `gpt-4o-mini`. "
    )

    with gr.Row():
        api_key_input = gr.Textbox(
            label="OpenAI API Key (Введіть вручну або він буде взятий із файлу .env, якщо такий існує)",
            placeholder="sk-proj-...",
            type="password"
        )
        dialect_input = gr.Dropdown(
            label="СУБД / Діалект",
            choices=["PostgreSQL", "T-SQL (MS SQL)", "MySQL", "SQLite"],
            value="PostgreSQL"
        )

    schema_input = gr.Code(
        label="Схема бази даних (DDL)",
        value=DEFAULT_SCHEMA,
        language="sql",
        lines=8
    )

    query_input = gr.Textbox(
        label="Опис запиту природною мовою",
        value=DEFAULT_QUERY,
        lines=3
    )

    submit_btn = gr.Button("Згенерувати SQL", variant="primary")

    sql_output = gr.Code(
        label="Згенерований SQL-код",
        language="sql",
        interactive=False
    )

    submit_btn.click(
        fn=generate_sql,
        inputs=[api_key_input, dialect_input, schema_input, query_input],
        outputs=sql_output
    )

if __name__ == "__main__":
    demo.launch()