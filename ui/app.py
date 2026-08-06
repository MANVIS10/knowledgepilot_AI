import os
import gradio as gr

from src.pipeline.knowledge_base import KnowledgeBase

# Don't create the KnowledgeBase during startup
kb = None


def chat(message, history):
    """
    Streams the assistant response.
    """

    global kb

    # Initialize only on the first request
    if kb is None:
        print("Initializing KnowledgeBase...")
        kb = KnowledgeBase()
        print("KnowledgeBase initialized.")

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


demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)