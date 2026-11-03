"""Вебінтерфейс AI Examiner на Gradio."""
from __future__ import annotations

import json
import os
import gradio as gr

from openai import OpenAI
from agent import GREETING, ExaminerAgent
from exam_tools import RESULTS_DIR

PROVIDERS = {
    "OpenAI": {"base_url": "https://api.openai.com/v1", "model": "gpt-4.1-mini"},
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
    },
    "Groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "OpenRouter": {"base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4.1-mini"},
    "Інший (OpenAI-сумісний)": {"base_url": "", "model": ""},
}
DEFAULT_PROVIDER = "OpenAI"

DESCRIPTION = """
# 🎓 AI Examiner — іспит з NLP

LLM-агент, який проводить усний іспит з курсу NLP у форматі чату.
Агент сам вирішує, коли викликати інструменти:
`start_exam` (перевірка студента й вибір тем), `get_next_topic`, `record_topic_result` і `end_exam` (запис результату).

**Як почати:** вкажіть API-ключ у налаштуваннях нижче і напишіть в чат ім'я, прізвище та email.
Для демо можна використати студента **Test Student**, `test@example.com`.
"""


def initial_chat() -> list[dict]:
    return [{"role": "assistant", "content": GREETING}]


def on_provider_change(provider: str):
    preset = PROVIDERS[provider]
    return preset["base_url"], preset["model"]


def add_user_message(text: str, chat: list[dict]):
    text = (text or "").strip()
    if not text:
        return "", chat, gr.update()
    return "", chat + [{"role": "user", "content": text}], text


def bot_turn(pending: str, chat, agent, api_key, base_url, model, show_tools):
    if not pending:
        return chat, agent, ""

    api_key = (api_key or "").strip() or os.getenv("LLM_API_KEY", "")
    if not api_key:
        chat = chat[:-1] + [
            {"role": "assistant", "content": "⚠️ Вкажіть API-ключ у налаштуваннях і надішліть повідомлення ще раз."}
        ]
        return chat, agent, ""

    if agent is not None and agent.session.finished:
        return chat + [
            {"role": "assistant", "content": "Іспит уже завершено. Щоб почати новий, натисніть «Новий іспит»."}
        ], agent, ""

    client = OpenAI(api_key=api_key, base_url=(base_url or "").strip() or None)
    if agent is None:
        agent = ExaminerAgent(client=client, model=model.strip())
    else:
        agent.client, agent.model = client, model.strip()

    try:
        reply, events = agent.respond(pending)
    except Exception as e:  # помилки мережі/API показуємо в чаті
        agent.rollback_last_user_message()
        return chat + [
            {"role": "assistant", "content": f"⚠️ Помилка LLM API: `{type(e).__name__}: {e}`"}
        ], agent, ""

    if show_tools:
        for ev in events:
            chat = chat + [{
                "role": "assistant",
                "content": f"**Аргументи:** `{json.dumps(ev.arguments, ensure_ascii=False)}`\n\n"
                           f"**Результат:** `{ev.result}`",
                "metadata": {"title": f"🔧 {ev.name}"},
            }]
    if reply:
        chat = chat + [{"role": "assistant", "content": reply}]
    if agent.session.finished:
        chat = chat + [{"role": "assistant", "content": "✅ Результат іспиту збережено."}]
    return chat, agent, ""


def new_exam():
    return initial_chat(), None, ""


def load_results():
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append([
            r.get("finished_at", ""),
            r.get("name", ""),
            r.get("email", ""),
            r.get("score", ""),
            "; ".join(r.get("topics", [])),
            r.get("feedback", ""),
        ])
    return rows


with gr.Blocks(title="AI Examiner") as demo:
    gr.Markdown(DESCRIPTION)
    agent_state = gr.State(None)
    pending_state = gr.State("")

    with gr.Tab("Іспит"):
        with gr.Accordion("⚙️ Налаштування LLM", open=True):
            with gr.Row():
                provider = gr.Dropdown(list(PROVIDERS), value=DEFAULT_PROVIDER, label="Провайдер")
                model = gr.Textbox(value=PROVIDERS[DEFAULT_PROVIDER]["model"], label="Модель")
            with gr.Row():
                api_key = gr.Textbox(label="API-ключ", type="password", placeholder="sk-...")
                base_url = gr.Textbox(value=PROVIDERS[DEFAULT_PROVIDER]["base_url"], label="Base URL")
            show_tools = gr.Checkbox(value=True, label="Показувати виклики інструментів у чаті")

        chatbot = gr.Chatbot(value=initial_chat(), height=520, label="Екзаменатор")
        with gr.Row():
            msg = gr.Textbox(placeholder="Ваша відповідь…", show_label=False, scale=8, autofocus=True)
            send = gr.Button("Надіслати", variant="primary", scale=1)
        reset = gr.Button("🔄 Новий іспит")

    with gr.Tab("Результати"):
        gr.Markdown("Збережені результати іспитів (файли в `results/`).")
        results_table = gr.Dataframe(
            headers=["Завершено", "Ім'я", "Email", "Оцінка", "Теми", "Відгук"],
            value=load_results,
            wrap=True,
        )
        gr.Button("Оновити").click(load_results, outputs=results_table)

    provider.change(on_provider_change, provider, [base_url, model])

    turn_inputs = [pending_state, chatbot, agent_state, api_key, base_url, model, show_tools]
    turn_outputs = [chatbot, agent_state, pending_state]
    for trigger in (msg.submit, send.click):
        trigger(add_user_message, [msg, chatbot], [msg, chatbot, pending_state]).then(
            bot_turn, turn_inputs, turn_outputs
        )
    reset.click(new_exam, outputs=[chatbot, agent_state, pending_state])


if __name__ == "__main__":
    demo.launch()