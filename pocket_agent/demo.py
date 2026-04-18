from __future__ import annotations

import argparse
from typing import Any


def chat(prompt: str, history: list[dict[str, Any]] | None = None) -> str:
    from inference import run

    return run(prompt, history or [])


def _history_to_messages(chat_history: list[dict[str, str]] | list[list[str]] | None) -> list[dict[str, str]]:
    if not chat_history:
        return []

    messages: list[dict[str, str]] = []
    for item in chat_history:
        if isinstance(item, dict):
            role = str(item.get("role", "user"))
            content = str(item.get("content", ""))
            if role in {"user", "assistant", "system"} and content:
                messages.append({"role": role, "content": content})
        elif isinstance(item, list) and len(item) == 2:
            user_text, assistant_text = item
            if isinstance(user_text, str) and user_text:
                messages.append({"role": "user", "content": user_text})
            if isinstance(assistant_text, str) and assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
    return messages


def _messages_to_history(messages: list[dict[str, str]]) -> list[list[str]]:
    history: list[list[str]] = []
    pending_user: str | None = None
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            history.append([pending_user, content])
            pending_user = None
    return history


def launch_gradio(share: bool = False) -> None:
    try:
        import gradio as gr  # type: ignore
    except Exception as exc:  # pragma: no cover - demo dependency optional
        raise SystemExit(f"Gradio is not available: {exc}")

    def respond(message: str, chat_history: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
        history_messages = _history_to_messages(_messages_to_history(chat_history))
        assistant_text = chat(message, history_messages)
        updated_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": assistant_text},
        ]
        return "", updated_history

    with gr.Blocks(title="Pocket-Agent Demo") as demo:
        gr.Markdown(
            "# Pocket-Agent\n"
            "Send a message and the model will reply with either a tool call or a refusal."
        )
        chatbot = gr.Chatbot(height=420, type="messages", allow_tags=False)
        with gr.Row():
            prompt = gr.Textbox(
                label="Message",
                placeholder="Ask for weather, calendar, conversion, currency, or SQL help.",
                scale=5,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        clear = gr.Button("Clear")

        prompt.submit(respond, [prompt, chatbot], [prompt, chatbot])
        send.click(respond, [prompt, chatbot], [prompt, chatbot])
        clear.click(lambda: ("", []), None, [prompt, chatbot])

    demo.launch(inbrowser=False, share=share)


def launch_cli() -> None:
    print("Pocket-Agent CLI. Type a message and press Enter. Blank line exits.")
    history: list[dict[str, str]] = []
    while True:
        prompt = input("you> ").strip()
        if not prompt:
            break
        answer = chat(prompt, history)
        print(f"assistant> {answer}")
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": answer})


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Pocket-Agent demo.")
    parser.add_argument("--cli", action="store_true", help="Run in terminal mode instead of Gradio.")
    parser.add_argument("--share", action="store_true", help="Enable a public Gradio link (recommended for Colab).")
    args = parser.parse_args()
    if args.cli:
        launch_cli()
    else:
        launch_gradio(share=args.share)


if __name__ == "__main__":
    main()
