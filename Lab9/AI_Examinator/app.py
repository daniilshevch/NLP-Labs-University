"""Вебінтерфейс AI Examiner на Gradio."""
from __future__ import annotations

import json
import os
import gradio as gr
from openai import OpenAI
from agent import GREETING, ExaminerAgent
from exam_tools import RESULTS_DIR
from student_agent import generate_student_reply, load_personas, load_students


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

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

# Персони AI-студента і студенти, від імені яких він складає іспит.
PERSONAS = {p["label"]: p for p in load_personas()}
DEFAULT_PERSONA = "Добре"
STUDENTS = {f"{s['name']} ({s['email']})": s for s in load_students()}
DEFAULT_STUDENT = next((k for k, s in STUDENTS.items() if s["name"] == "Test Student"), next(iter(STUDENTS)))

DESCRIPTION = """
# 🎓 AI Examiner — іспит з NLP

LLM-агент, який проводить усний іспит з курсу NLP у форматі чату.
Агент сам вирішує, коли викликати інструменти:
`start_exam` (перевірка студента й вибір тем), `get_next_topic`, `record_topic_result` і `end_exam` (запис результату).

**Як почати:** вкажіть API-ключ у налаштуваннях нижче і напишіть в чат ім'я, прізвище та email.
Для демо можна використати студента **Test Student**, `test@lpnu.ua`.

**AI-студент:** на будь-якому кроці замість ручної відповіді можна натиснути «Згенерувати відповідь».
Модель-студент напише відповідь відповідно до обраного рівня знань, і перед відправкою її можна відредагувати.
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
    except Exception as e:
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


def generate_answer(agent, api_key, base_url, model, student_model, persona_label, student_label):
    """Генерує відповідь AI-студента і кладе її в поле вводу."""
    if agent is not None and agent.session.finished:
        raise gr.Error("Іспит уже завершено. Щоб почати новий, натисніть «Новий іспит».")
    api_key = (api_key or "").strip() or os.getenv("LLM_API_KEY", "")
    if not api_key:
        raise gr.Error("Вкажіть API-ключ у налаштуваннях LLM.")

    # Студент бачить те саме, що бачила б людина: журнал іспиту без викликів інструментів.
    transcript = agent.log if agent is not None else [{"role": "system", "content": GREETING, "datetime": ""}]
    client = OpenAI(api_key=api_key, base_url=(base_url or "").strip() or None)
    try:
        text = generate_student_reply(
            client=client,
            model=(student_model or "").strip() or model.strip(),
            persona_prompt=PERSONAS[persona_label]["prompt"],
            student=STUDENTS[student_label],
            transcript=transcript,
        )
    except Exception as e:  # помилки мережі/API
        raise gr.Error(f"Не вдалося згенерувати відповідь: {type(e).__name__}: {e}")
    if not text:
        raise gr.Error("Модель повернула порожню відповідь. Спробуйте ще раз.")
    return text


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

        with gr.Accordion("🤖 AI-студент", open=False):
            gr.Markdown(
                "Модель-студент використовує того самого провайдера і ключ, що й екзаменатор. "
                "Модель краще взяти іншу: модель схильна завищувати оцінку відповідям у власному стилі."
            )
            with gr.Row():
                persona = gr.Dropdown(list(PERSONAS), value=DEFAULT_PERSONA, label="Рівень відповідей")
                student = gr.Dropdown(list(STUDENTS), value=DEFAULT_STUDENT, label="Від імені студента")
                student_model = gr.Textbox(label="Модель студента", placeholder="порожньо = як в екзаменатора")

        chatbot = gr.Chatbot(value=initial_chat(), height=520, label="Екзаменатор")
        with gr.Row():
            msg = gr.Textbox(placeholder="Ваша відповідь…", show_label=False, scale=8, autofocus=True)
            send = gr.Button("Надіслати", variant="primary", scale=1)
        with gr.Row():
            generate = gr.Button("🤖 Згенерувати відповідь")
            generate_send = gr.Button("🤖 Згенерувати й надіслати")
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
    # AI-студент: «Згенерувати» лише заповнює поле вводу, «Згенерувати й надіслати»
    # після цього запускає звичайний хід. .success() не запускає хід, якщо генерація впала.
    gen_inputs = [agent_state, api_key, base_url, model, student_model, persona, student]
    generate.click(generate_answer, gen_inputs, msg)
    generate_send.click(generate_answer, gen_inputs, msg).success(
        add_user_message, [msg, chatbot], [msg, chatbot, pending_state]
    ).then(bot_turn, turn_inputs, turn_outputs)

    reset.click(new_exam, outputs=[chatbot, agent_state, pending_state])


if __name__ == "__main__":
    demo.launch(ssr_mode=False)