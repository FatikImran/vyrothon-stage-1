from __future__ import annotations

import argparse
from typing import Any


def chat(prompt: str, history: list[dict[str, Any]] | None = None) -> str:
    from inference import run

    return run(prompt, history or [])


def launch_gradio() -> None:
    try:
        import gradio as gr  # type: ignore
    except Exception as exc:  # pragma: no cover - demo dependency optional
        raise SystemExit(f"Gradio is not available: {exc}")

    def respond(message: str, chat_history: list[list[str]]) -> tuple[str, list[list[str]]]:
        history_messages: list[dict[str, str]] = []
        for user_text, assistant_text in chat_history:
            history_messages.append({"role": "user", "content": user_text})
            history_messages.append({"role": "assistant", "content": assistant_text})
        assistant_text = chat(message, history_messages)
        chat_history = chat_history + [[message, assistant_text]]
        return "", chat_history

    with gr.Blocks(title="Pocket-Agent Demo") as demo:
        gr.Markdown("# Pocket-Agent\nOffline tool-calling demo with visible tool-call output.")
        chatbot = gr.Chatbot(height=420)
        prompt = gr.Textbox(label="Message", placeholder="Ask for weather, calendar, conversion, currency, or SQL help.")
        clear = gr.Button("Clear")
        prompt.submit(respond, [prompt, chatbot], [prompt, chatbot])
        clear.click(lambda: ("", []), None, [prompt, chatbot])
    demo.launch()


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
    args = parser.parse_args()
    if args.cli:
        launch_cli()
    else:
        launch_gradio()


if __name__ == "__main__":
    main()
