from __future__ import annotations

import gradio as gr

from inference import run


def _to_history_messages(chat_history: list[tuple[str, str]] | None) -> list[dict[str, str]]:
    if not chat_history:
        return []
    history: list[dict[str, str]] = []
    for user_text, assistant_text in chat_history:
        if user_text:
            history.append({"role": "user", "content": user_text})
        if assistant_text:
            history.append({"role": "assistant", "content": assistant_text})
    return history


def respond(message: str, chat_history: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]]]:
    message = (message or "").strip()
    if not message:
        return "", chat_history

    answer = run(message, _to_history_messages(chat_history))
    updated_history = chat_history + [(message, answer)]
    return "", updated_history


with gr.Blocks(title="Pocket-Agent Demo") as demo:
    gr.Markdown("# Pocket-Agent\nSend messages and view tool-call responses.")
    chatbot = gr.Chatbot(height=460)
    with gr.Row():
        textbox = gr.Textbox(label="Message", placeholder="Try weather, calendar, convert, currency, or SQL requests.", scale=5)
        send = gr.Button("Send", variant="primary", scale=1)
    clear = gr.Button("Clear")

    textbox.submit(respond, [textbox, chatbot], [textbox, chatbot])
    send.click(respond, [textbox, chatbot], [textbox, chatbot])
    clear.click(lambda: ("", []), None, [textbox, chatbot])


if __name__ == "__main__":
    demo.launch()
