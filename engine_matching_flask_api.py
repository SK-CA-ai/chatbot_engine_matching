"""Flask API for engine_matching helpers."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request
import google.genai as genai
from google.genai.errors import ClientError as GeminiClientError
from google.genai.errors import ServerError as GeminiServerError
from openai import OpenAI

from engine_matching import (
    build_product_enquiry_prompt,
    detect_escalation,
    engine_match,
    find_relevant_history_reply,
    summarize_conversation,
)
from excel_utils import load_knowledge_base
from store_locator import (
    build_store_locator_prompt,
    detect_language,
    detect_location,
    find_matching_stores,
    is_location_query,
    load_stores,
)
from faq_handler import is_faq_query, run_faq_lookup

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DEFAULT_KNOWLEDGE_PATH = Path(
    os.getenv("ENGINE_MATCHING_KB_PATH", BASE_DIR / "data" / "Samples.xlsx")
)
DEFAULT_KNOWLEDGE_SHEET = os.getenv("ENGINE_MATCHING_KB_SHEET", "Main DB")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

app = Flask(__name__)


@app.errorhandler(GeminiClientError)
def handle_gemini_client_error(exc: GeminiClientError):
    """Catch Gemini 403 (leaked/invalid key) globally so it never returns 500."""
    status = getattr(exc, "status_code", 403)
    msg = (
        "The Gemini API key is invalid or has been revoked. "
        "Please update GEMINI_API_KEY in your environment settings."
    )
    print(f"⚠️ Gemini ClientError {status}: {exc}")
    return jsonify({"error": msg, "error_code": "INVALID_API_KEY"}), 503


_knowledge_df: pd.DataFrame | None = None


def _get_knowledge_df(
    knowledge_path: str | Path | None = None,
    knowledge_sheet: str | None = None,
) -> pd.DataFrame:
    global _knowledge_df

    path = Path(knowledge_path) if knowledge_path else DEFAULT_KNOWLEDGE_PATH
    sheet = knowledge_sheet or DEFAULT_KNOWLEDGE_SHEET

    if path != DEFAULT_KNOWLEDGE_PATH or sheet != DEFAULT_KNOWLEDGE_SHEET:
        return load_knowledge_base(path, sheet)

    if _knowledge_df is None:
        _knowledge_df = load_knowledge_base(path, sheet)

    return _knowledge_df


def _detect_emotion(text: str, provider: str) -> str:
    """Infer user emotion using an LLM instead of keyword rules."""

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""

    prompt = (
        "You are an emotion detector for a customer support chatbot. "
        "Classify the user's primary emotion as one of: frustrated, worried, confused, sad, or neutral "
        "(use neutral when no clear emotion is present). "
        "Respond ONLY with a JSON object like {\"emotion\": \"frustrated\"}. "
        "Do not add explanations.\n"
        f"User message: \"{cleaned}\""
    )

    provider_name = (provider or "").lower() or os.getenv("MODEL_PROVIDER", "gemini").lower()
    try:
        if provider_name == "openai":
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=DEFAULT_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Return only a JSON object with an 'emotion' field."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
        else:
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model=DEFAULT_GEMINI_MODEL,
                contents=prompt,
            )
            content = response.text

        parsed = json.loads(content)
        emotion = str(parsed.get("emotion", "")).lower()
    except Exception as exc:  # pragma: no cover - defensive fallback for runtime failures
        print("⚠️ Emotion detection failed:", exc)
        return ""

    valid_emotions = {"frustrated", "worried", "confused", "sad"}
    return emotion if emotion in valid_emotions else ""


def _extract_keywords(text: str) -> set[str]:
    """Return a set of normalized keywords from ``text`` for overlap checks."""

    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    return {token for token in tokens if len(token) > 2}


def _history_reply_by_keyword(
    conversation_history: list[str], current_question: str
) -> str | None:
    """Find the best matching history entry based on shared keywords."""

    trimmed_question = current_question.strip().lower()
    history = [entry for entry in conversation_history if str(entry).strip()]
    if history and trimmed_question and history[-1].strip().lower() == trimmed_question:
        history = history[:-1]

    keywords = _extract_keywords(current_question)
    if not history or not keywords:
        return None

    best_entry: str | None = None
    best_score = 0
    for entry in reversed(history):
        score = len(keywords & _extract_keywords(entry))
        if score > best_score:
            best_entry = entry.strip()
            best_score = score

    return best_entry if best_score > 0 else None


def _build_sales_redirect_prompt(user_message: str, product_json: str) -> str:
    return (
        "You are a friendly CompAsia sales consultant. The user asked an unrelated question. "
        "Respond politely, then pivot to highlight CompAsia's refurbished HP devices and services. "
        "Keep it concise, sound natural, and ask one gentle follow-up question based on the user's message.\n\n"
        "Available products (JSON):\n"
        f"{product_json}\n\n"
        f"User message: \"{user_message}\""
    )


def _generate_store_reply(prompt: str, provider: str) -> str:
    """Call the LLM to turn a store locator prompt into a friendly reply."""
    provider_name = (provider or "").lower()
    try:
        if provider_name == "openai":
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=DEFAULT_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Reply as a friendly retail assistant in plain text."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        else:
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            try:
                response = client.models.generate_content(
                    model=DEFAULT_GEMINI_MODEL,
                    contents=prompt,
                )
            except GeminiServerError as exc:
                if exc.status_code != 503:
                    raise
                print(f"⚠️ {DEFAULT_GEMINI_MODEL} returned 503, retrying with {DEFAULT_GEMINI_MODEL.replace('2.5', '2.0')}")
                response = client.models.generate_content(
                    model=DEFAULT_GEMINI_MODEL.replace("2.5", "2.0"),
                    contents=prompt,
                )
            return response.text.strip()
    except Exception as exc:
        print("⚠️ Store locator LLM call failed:", exc)
        return ""


def _run_store_locator(question: str, provider: str) -> dict:
    """
    Detect location in *question*, query the store CSV, generate a reply.
    Returns a dict ready to embed in any endpoint response.
    """
    language = detect_language(question)
    location_term = detect_location(question)

    if location_term is None:
        reply = (
            "Untuk cari kedai berdekatan, boleh beritahu kawasan atau bandar anda?"
            if language == "ms"
            else "Sure! To find the nearest store for you, could you share your current area or city?"
        )
        return {
            "needs_location": True,
            "stores": [],
            "closed_stores": [],
            "reply": reply,
            "language": language,
        }

    active_stores, closed_stores = find_matching_stores(location_term, load_stores())

    if not active_stores and not closed_stores:
        reply = (
            "Maaf, tiada kedai ditemui berhampiran kawasan tersebut. "
            "Cuba semak kawasan berhampiran seperti Kuala Lumpur, Selangor, Penang, atau Johor."
            if language == "ms"
            else "Sorry, no stores were found near that area. "
               "Try checking nearby areas such as Kuala Lumpur, Selangor, Penang, or Johor."
        )
        return {
            "needs_location": False,
            "stores": [],
            "closed_stores": [],
            "reply": reply,
            "language": language,
            "location_detected": location_term,
        }

    prompt = build_store_locator_prompt(question, active_stores, language)
    reply = _generate_store_reply(prompt, provider)
    if not reply:
        # plain-text fallback if LLM fails
        reply = "\n\n".join(
            f"📍 {s['name']}\n🏢 {s['location']}\n🕐 {s['operatingHours']}\n💬 {s['whatsappLink']}"
            for s in active_stores[:5]
        )

    return {
        "needs_location": False,
        "stores": active_stores,
        "closed_stores": closed_stores,
        "reply": reply,
        "language": language,
        "location_detected": location_term,
    }


@app.get("/")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, 200


@app.post("/detect-escalation")
def detect_escalation_endpoint() -> tuple[Any, int]:
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "")

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Question cannot be empty."}), 400

    should_escalate, response = detect_escalation(question)
    return jsonify({"escalate": should_escalate, "response": response}), 200


@app.post("/detect-emotion")
def detect_emotion_endpoint() -> tuple[Any, int]:
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    provider = payload.get("provider", "gemini")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Text cannot be empty."}), 400

    emotion = _detect_emotion(text, provider)
    return jsonify({"emotion": emotion}), 200


@app.post("/engine-match")
def engine_match_endpoint() -> tuple[Any, int]:
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "")
    provider = payload.get("provider", "gemini")
    conversation_summary = payload.get("conversation_summary", "")
    print("HelloConversationSummary: ", conversation_summary)
    stock_table_schema = payload.get("stock_table_schema", "")
    if not stock_table_schema:
        stock_table_schema = payload.get("iphone_stock_json", "")
    knowledge_path = payload.get("knowledge_path")
    knowledge_sheet = payload.get("knowledge_sheet")

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Question cannot be empty."}), 400

    # --- Store locator intercept ---
    # If the question is about finding a physical store, skip the KB match
    # and return store results directly under match = "STORE_LOCATOR".
    if is_location_query(question):
        store_result = _run_store_locator(question, provider)
        match_key = (
            "STORE_LOCATOR_NEEDS_LOCATION"
            if store_result["needs_location"]
            else "STORE_LOCATOR"
        )
        store_reply = store_result.get("reply", "")
        return jsonify({
            "match": match_key,
            "score": 1.0,
            # Populate matched_row.answer so chatbots that read matched_row
            # (instead of the top-level reply) still receive the store reply.
            "matched_row": {"keyword": match_key, "answer": store_reply},
            "reply": store_reply,
            "store_locator": store_result,
        }), 200
    # --- End store locator intercept ---

    # --- FAQ intercept ---
    # If the question matches a CompAsia FAQ topic, look up the answer from
    # CompAsia_FAQ.docx and return it directly — no KB match needed.
    if is_faq_query(question):
        gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if provider.lower() == "openai" else None
        faq_result = run_faq_lookup(
            question, provider,
            gemini_client=gemini_client,
            openai_client=openai_client,
            gemini_model=DEFAULT_GEMINI_MODEL,
            openai_model=DEFAULT_OPENAI_MODEL,
        )
        faq_reply = faq_result.get("reply", "")
        return jsonify({
            "match": "FAQ",
            "score": 1.0,
            "matched_row": {"keyword": "FAQ", "answer": faq_reply},
            "reply": faq_reply,
            "faq": faq_result,
        }), 200
    # --- End FAQ intercept ---

    knowledge_df = _get_knowledge_df(knowledge_path, knowledge_sheet)
    try:
        match, score, matched_row = engine_match(
            question,
            knowledge_df,
            provider=provider,
            conversation_summary=conversation_summary,
            stock_table_schema=stock_table_schema,
        )
    except GeminiServerError as exc:
        if exc.status_code == 503:
            return jsonify({
                "error": "AI model is temporarily unavailable due to high demand. Please try again in a moment.",
                "error_code": "MODEL_UNAVAILABLE",
            }), 503
        raise

    if isinstance(matched_row, pd.Series):
        matched_payload: Any = matched_row.to_dict()
    else:
        matched_payload = matched_row

    return (
        jsonify({
            "match": match,
            "score": score,
            "matched_row": matched_payload,
        }),
        200,
    )


@app.post("/history-reply-keyword")
def history_reply_keyword_endpoint() -> tuple[Any, int]:
    payload = request.get_json(silent=True) or {}
    conversation_history = payload.get("conversation_history", [])
    question = payload.get("question", "")

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Question cannot be empty."}), 400
    if not isinstance(conversation_history, list):
        return jsonify({"error": "conversation_history must be a list."}), 400

    reply = _history_reply_by_keyword(conversation_history, question)
    return jsonify({"reply": reply}), 200


@app.post("/summarize")
def summarize_endpoint() -> tuple[Any, int]:
    payload = request.get_json(silent=True) or {}
    conversation_history = payload.get("conversation_history", [])
    question = payload.get("question", "")
    answer = payload.get("answer", "")
    provider = payload.get("provider", "gemini")
    previous_summary = payload.get("previous_summary", "")

    if not isinstance(conversation_history, list):
        return jsonify({"error": "conversation_history must be a list."}), 400
    if question and not isinstance(question, str):
        return jsonify({"error": "question must be a string."}), 400
    if answer and not isinstance(answer, str):
        return jsonify({"error": "answer must be a string."}), 400

    if not conversation_history and (question.strip() or answer.strip()):
        conversation_history = []
        if question.strip():
            conversation_history.append(f"Customer: {question.strip()}")
        if answer.strip():
            conversation_history.append(f"Agent: {answer.strip()}")

    summary = summarize_conversation(
        conversation_history,
        provider=provider,
        previous_summary=previous_summary,
    )
    return jsonify({"summary": summary}), 200


@app.post("/history-reply")
def history_reply_endpoint() -> tuple[Any, int]:
    payload = request.get_json(silent=True) or {}
    conversation_history = payload.get("conversation_history", [])
    question = payload.get("question", "")
    provider = payload.get("provider", "gemini")

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Question cannot be empty."}), 400
    if not isinstance(conversation_history, list):
        return jsonify({"error": "conversation_history must be a list."}), 400

    reply = find_relevant_history_reply(
        conversation_history,
        question,
        provider=provider,
    )
    return jsonify({"reply": reply}), 200


@app.post("/product-prompt")
def product_prompt_endpoint() -> tuple[Any, int]:
    payload = request.get_json(silent=True) or {}
    user_message = payload.get("user_message", "")
    stock_table_schema = payload.get("stock_table_schema", "")
    if not stock_table_schema:
        stock_table_schema = payload.get("iphone_stock_json", "")
    conversation_summary = payload.get("conversation_summary", "")

    if not isinstance(user_message, str) or not user_message.strip():
        return jsonify({"error": "user_message cannot be empty."}), 400

    prompt = build_product_enquiry_prompt(
        user_message,
        stock_table_schema,
        conversation_summary=conversation_summary,
    )
    return jsonify({"prompt": prompt}), 200


@app.post("/sales-redirect")
def sales_redirect_endpoint() -> tuple[Any, int]:
    payload = request.get_json(silent=True) or {}
    user_message = payload.get("user_message", "")
    provider = payload.get("provider", "gemini")
    product_json = payload.get("product_json", "")

    if not isinstance(user_message, str) or not user_message.strip():
        return jsonify({"error": "user_message cannot be empty."}), 400

    prompt = _build_sales_redirect_prompt(user_message, product_json)
    provider_name = (provider or "").lower()

    if provider_name == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Reply as a friendly sales consultant in plain text."},
                {"role": "user", "content": prompt},
            ],
        )
        reply = response.choices[0].message.content.strip()
    else:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=DEFAULT_GEMINI_MODEL,
            contents=prompt,
        )
        reply = response.text.strip()

    return jsonify({"reply": reply}), 200


@app.post("/store-locator")
def store_locator_endpoint() -> tuple[Any, int]:
    payload = request.get_json(silent=True) or {}
    user_message = payload.get("user_message", "")
    provider = payload.get("provider", "gemini")

    if not isinstance(user_message, str) or not user_message.strip():
        return jsonify({"error": "user_message cannot be empty."}), 400

    return jsonify(_run_store_locator(user_message, provider)), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5050"))
    app.run(host="0.0.0.0", port=port)
