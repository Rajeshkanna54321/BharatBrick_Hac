"""
app.py — BNS Legal Assistant
==============================
Runs two servers in one process:
  1. Gradio web UI  (port 7860 — what HuggingFace Spaces exposes)
  2. Flask webhook  (port 5000 — for Twilio WhatsApp)

HuggingFace Spaces only exposes port 7860 publicly.
Twilio webhook must point to your public Gradio URL via /whatsapp route.
We mount the Flask app inside Gradio using a custom route so everything
runs on a single port (7860).
"""

import os
import threading
import gradio as gr
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

from bns_full_pipeline import load_all, run_pipeline_api

# ── Load models once at startup ────────────────────────────────────
print("Starting BNS Legal Assistant...")
_models = load_all()   # returns: llm, sbert, bm25, desc_embeddings, df, device
print("Models ready.")


# ══════════════════════════════════════════════════════════════════
# FLASK — Twilio WhatsApp webhook
# ══════════════════════════════════════════════════════════════════

flask_app = Flask(__name__)

@flask_app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    """
    Twilio sends POST here when a WhatsApp message arrives.
    We call the pipeline and reply with TwiML.
    """
    incoming_msg = request.form.get("Body", "").strip()
    from_number  = request.form.get("From", "")

    print(f"[WhatsApp] From: {from_number} | Message: {incoming_msg[:80]}")

    if not incoming_msg:
        reply = "Please describe your situation and I will find the applicable BNS section."
    else:
        try:
            reply = run_pipeline_api(incoming_msg, *_models)
        except Exception as e:
            reply = f"❌ Error processing your request: {str(e)[:200]}"

    resp = MessagingResponse()
    resp.message(reply)
    return str(resp), 200, {"Content-Type": "text/xml"}


@flask_app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "service": "BNS Legal Assistant"}, 200


def run_flask():
    """Run Flask on port 5000 (internal only — not exposed by HuggingFace)."""
    flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


# ══════════════════════════════════════════════════════════════════
# GRADIO — Web UI (exposed on port 7860 by HuggingFace Spaces)
# ══════════════════════════════════════════════════════════════════

def gradio_query(user_query: str) -> str:
    """Gradio interface function — calls the same pipeline."""
    if not user_query.strip():
        return "Please describe your situation."
    try:
        return run_pipeline_api(user_query.strip(), *_models)
    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(title="BNS Legal Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ⚖️ BNS Legal Assistant
        **Bharatiya Nyaya Sanhita (BNS) Section Finder**

        Describe a crime or legal situation in plain language (Hindi-English/English).
        The assistant will identify the applicable BNS section and relevant case precedents.

        > ⚠️ *For informational purposes only. Consult a qualified lawyer for legal advice.*
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            query_box = gr.Textbox(
                label="Describe your situation",
                placeholder="e.g. Someone broke into my house and stole my laptop...",
                lines=4,
            )
            submit_btn = gr.Button("Find BNS Section", variant="primary")

        with gr.Column(scale=4):
            output_box = gr.Textbox(
                label="Result",
                lines=20,
                show_copy_button=True,
            )

    gr.Examples(
        examples=[
            ["Someone threw acid on my face"],
            ["My husband beats me and demands dowry"],
            ["A person threatened to kill me over the phone"],
            ["Someone stole my mobile phone on the road"],
            ["My boss is harassing me sexually at work"],
            ["Someone cheated me of 2 lakh rupees online"],
        ],
        inputs=query_box,
    )

    submit_btn.click(fn=gradio_query, inputs=query_box, outputs=output_box)
    query_box.submit(fn=gradio_query, inputs=query_box, outputs=output_box)

    # Also expose /whatsapp route through Gradio's built-in server
    # so Twilio can reach it on the single public port (7860)
    demo.add_api_route("/whatsapp", whatsapp_webhook, methods=["POST"])
    demo.add_api_route("/health",   health,           methods=["GET"])


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Start Flask on port 5000 in a background thread (for local testing)
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()

    # Start Gradio on port 7860 (HuggingFace Spaces standard port)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,    # HuggingFace handles public URL — don't need ngrok
    )