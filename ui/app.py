import gradio as gr

from src.pipeline.knowledge_base import KnowledgeBase

kb = KnowledgeBase()


def chat(message, history):
    """
    Streams the assistant response.
    """

    for partial in kb.stream_answer(message):
        yield partial


demo = gr.ChatInterface(
    fn=chat,
    title="🧠 KnowledgePilot",
    description="""
Ask questions about the Stanford CS229 course.

Examples:
• What is Logistic Regression?
• Explain Gradient Descent.
• What is Reinforcement Learning?
""",
    examples=[
        "What is Logistic Regression?",
        "Explain Gradient Descent.",
        "What is Reinforcement Learning?",
        "Explain Support Vector Machines."
    ]
)

import os

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)