import os
import gradio as gr

from src.pipeline.knowledge_base import KnowledgeBase

# Don't create the KnowledgeBase during startup
kb = None

def chat(message, history, debug_mode, request: gr.Request):
    """
    Streams the assistant response and updates sources & debug panels.
    """
    global kb

    # Initialize only on the first request
    if kb is None:
        print("Initializing KnowledgeBase...")
        kb = KnowledgeBase()
        print("KnowledgeBase initialized.")

    history = history or []
    # Add the user message and an initial empty assistant message to history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "⚡ Searching database and thinking..."})

    yield history, "🔍 Retrieving relevant lecture sources...", "⚙️ Initializing vector similarity search..."

    try:
        sources_md = "No sources cited for this response."
        debug_md = "Debug Mode is disabled. Enable 'Retrieval-Debug Mode' below the chat and ask a question to see details."

        stream = kb.stream_answer(message, session_id=request.session_hash)
        
        # Get the first yielded item which contains the retrieval metadata
        first_update = next(stream)
        
        # Format the sources markdown
        sources_list = first_update.get("sources", [])
        if sources_list:
            sources_md_parts = []
            for i, src in enumerate(sources_list, start=1):
                pct = int(src["similarity_score"] * 100)
                badge_class = "badge-high" if pct >= 80 else ("badge-med" if pct >= 60 else "badge-low")
                sources_md_parts.append(
                    f'<details class="source-card">'
                    f'  <summary><b>Source [{i}]</b>: {src["lecture"]}, Section {src["chunk_id"]} '
                    f'    <span class="badge {badge_class}">Similarity: {pct}%</span>'
                    f'  </summary>'
                    f'  <div class="source-content">{src["text"]}</div>'
                    f'</details>'
                )
            sources_md = "\n".join(sources_md_parts)
        else:
            sources_md = "No sources cited for this response."

        # Format the debug markdown if debug mode is active
        if debug_mode:
            ret_chunks = first_update.get("retrieved_chunks", [])
            passed_chunks = first_update.get("passed_chunks", [])
            
            debug_parts = []
            debug_parts.append(f"### 🔍 Search Query\n`{message}`\n")
            debug_parts.append(f"### 📊 Guardrails & Retrieval Status\n")
            status_badge = '<span class="badge badge-high">Success</span>' if first_update.get("success", True) else '<span class="badge badge-low">Guardrail Triggered</span>'
            debug_parts.append(f"- **Retrieval Guard**: {status_badge}\n")
            debug_parts.append(f"- **Top-K Chunks Retrieved**: {len(ret_chunks)}\n")
            debug_parts.append(f"- **Total Chunks Passed to LLM**: {len(passed_chunks)}\n")
            
            debug_parts.append(f"\n### 📚 All Retrieved Chunks Details\n")
            for i, chunk in enumerate(ret_chunks, start=1):
                pct = int(chunk["similarity_score"] * 100)
                dist = chunk["distance"]
                passed = "Yes" if any(c["chunk_id"] == chunk["chunk_id"] and c["lecture"] == chunk["lecture"] for c in passed_chunks) else "No"
                passed_badge = '<span class="badge badge-high">Yes</span>' if passed == "Yes" else '<span class="badge badge-low">No</span>'
                badge_class = "badge-high" if pct >= 80 else ("badge-med" if pct >= 60 else "badge-low")
                
                debug_parts.append(
                    f'<details class="source-card" style="border-left: 4px solid #6366f1;">'
                    f'  <summary><b>Rank {i}</b>: {chunk["lecture"].replace("lecture", "Lecture ")}, Section {chunk["chunk_id"]}'
                    f'    <span class="badge {badge_class}">{pct}% Similarity</span>'
                    f'    <span class="badge badge-info">Distance: {dist:.4f}</span>'
                    f'    <span class="badge badge-info">Passed: {passed}</span>'
                    f'  </summary>'
                    f'  <div class="source-content">'
                    f'    <p><b>Passed to LLM:</b> {passed_badge}</p>'
                    f'    <p><b>Raw Text:</b></p>'
                    f'    {chunk["text"]}'
                    f'  </div>'
                    f'</details>'
                )
            debug_md = "\n".join(debug_parts)
        else:
            debug_md = "Debug Mode is disabled. Enable 'Retrieval-Debug Mode' below the chat and ask a question to see details."

        # Start streaming the answer
        for update in stream:
            ans = update["answer"]
            history[-1]["content"] = ans
            yield history, sources_md, debug_md

    except Exception as e:
        history[-1]["content"] = f"An error occurred: {str(e)}"
        yield history, "Error loading sources.", f"Error detail: {str(e)}"


def clear_chat(request: gr.Request):
    """
    Clears the chatbot history and resets conversation memory.
    """
    global kb
    if kb is not None:
        kb.clear_memory(request.session_hash)
    return [], "No sources cited yet.", "Debug Mode is disabled. Enable 'Retrieval-Debug Mode' below the chat and ask a question to see details."


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

body, html, .gradio-container {
    font-family: 'Plus Jakarta Sans', 'Outfit', -apple-system, sans-serif !important;
    background-color: #0b0f19 !important;
    color: #f1f5f9 !important;
}

/* Header styling */
.header {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.4) 0%, rgba(49, 16, 66, 0.4) 100%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 24px 20px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
}

.header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 8px 0;
    background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}

.header p {
    font-size: 1.0rem;
    color: #94a3b8;
    margin: 0;
    font-weight: 400;
}

/* Tabs styling */
.tabs {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    background-color: #111827 !important;
    overflow: hidden;
}

/* Source card styling */
details.source-card {
    background-color: #1e293b;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    margin-bottom: 12px;
    padding: 0;
    overflow: hidden;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

details.source-card:hover {
    border-color: #6366f1;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15);
    transform: translateY(-1px);
}

details.source-card summary {
    padding: 12px 16px;
    font-weight: 600;
    cursor: pointer;
    outline: none;
    user-select: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: #f8fafc;
    background-color: #1e293b;
    transition: background-color 0.2s;
}

details.source-card summary:hover {
    background-color: #334155;
}

details.source-card[open] {
    border-color: #4f46e5;
}

details.source-card[open] summary {
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    background-color: #0f172a;
}

details.source-card .source-content {
    padding: 16px;
    font-size: 0.92rem;
    color: #cbd5e1;
    background-color: #0f172a;
    line-height: 1.6;
    max-height: 250px;
    overflow-y: auto;
    white-space: pre-wrap;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 6px;
    letter-spacing: 0.1px;
}

.badge-high {
    background-color: rgba(16, 185, 129, 0.12) !important;
    color: #34d399 !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
}

.badge-med {
    background-color: rgba(245, 158, 11, 0.12) !important;
    color: #fbbf24 !important;
    border: 1px solid rgba(245, 158, 11, 0.3) !important;
}

.badge-low {
    background-color: rgba(239, 68, 68, 0.12) !important;
    color: #f87171 !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
}

.badge-info {
    background-color: rgba(99, 102, 241, 0.12) !important;
    color: #818cf8 !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
}

/* Buttons styling */
.primary-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}

.primary-btn:hover {
    filter: brightness(1.15) !important;
}
"""

with gr.Blocks(title="KnowledgePilot - CS229 TA") as demo:
    # Page Header
    gr.HTML(
        """
        <div class="header">
            <h1>🧠 KnowledgePilot</h1>
            <p>Stanford CS229 Machine Learning Course Assistant with Grounded Citations</p>
        </div>
        """
    )
    
    with gr.Row():
        # Chat interface (Left Column)
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation History", 
                elem_id="chatbot"
            )
            msg = gr.Textbox(
                placeholder="Ask about Logistic Regression, SVM, Gradient Descent, etc...",
                label="Your Question",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary", elem_classes=["primary-btn"])
                clear_btn = gr.Button("Clear Chat")
                debug_mode = gr.Checkbox(
                    label="Retrieval-Debug Mode", 
                    value=False,
                    info="Show raw chunks, similarity scores, and LLM input details"
                )
                
            gr.Examples(
                examples=[
                    "What is Logistic Regression?",
                    "Explain Gradient Descent.",
                    "What is Reinforcement Learning?",
                    "Explain Support Vector Machines."
                ],
                inputs=msg,
                label="Example Questions"
            )
            
        # Citations & Debug (Right Column)
        with gr.Column(scale=2):
            with gr.Tabs(elem_classes=["tabs"]):
                with gr.Tab("Sources", id="sources_tab"):
                    sources_display = gr.Markdown(
                        "No sources cited yet. Ask a question to see supporting evidence.",
                        sanitize_html=False
                    )
                with gr.Tab("Retrieval Debug", id="debug_tab"):
                    debug_display = gr.Markdown(
                        "Debug Mode is disabled. Enable 'Retrieval-Debug Mode' below the chat and ask a question to see details.",
                        sanitize_html=False
                    )

    # Event handlers
    def start_interaction():
        return gr.update(interactive=False)

    def stop_interaction():
        return gr.update(interactive=True)

    # Submit event handlers
    msg_submit = msg.submit(
        fn=chat,
        inputs=[msg, chatbot, debug_mode],
        outputs=[chatbot, sources_display, debug_display]
    )
    msg_submit.then(fn=lambda: "", inputs=None, outputs=msg)

    btn_submit = submit_btn.click(
        fn=chat,
        inputs=[msg, chatbot, debug_mode],
        outputs=[chatbot, sources_display, debug_display]
    )
    btn_submit.then(fn=lambda: "", inputs=None, outputs=msg)

    # Clear event handler
    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[chatbot, sources_display, debug_display]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        css=custom_css
    )