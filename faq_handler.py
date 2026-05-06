"""FAQ handler — parse CompAsia_FAQ.docx and answer customer questions."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
FAQ_DOCX = BASE_DIR / "CompAsia_FAQ.docx"
FAQ_MD = BASE_DIR / ".claude" / "skills" / "compasia-faq" / "references" / "faq_content.md"

# ---------------------------------------------------------------------------
# Detection keywords — broad topics covered by the FAQ
# ---------------------------------------------------------------------------
_STRONG_FAQ_SIGNALS: set[str] = {
    # grading
    "grading", "grade", "asnew", "cosmetic",
    # warranty
    "warranty", "warrantee", "defect", "claim",
    # payments / installment
    "payment", "installment", "instalment", "paylater", "spaylater", "fpx",
    "visa", "mastercard",
    # programs
    "renewngo", "renew",
    # replace / sell
    "replace", "sell", "selling",
    # delivery
    "delivery", "shipping", "dispatch",
    # account
    "password", "login",
    # specific topics
    "imei", "tracking", "refurbished", "secondhand",
    "cancel", "cancellation",
}

_WEAK_FAQ_SIGNALS: set[str] = {
    "pay", "ship", "arrive", "deliver",
    "repair", "fix", "broken", "damage",
    "battery", "health", "condition",
    "upgrade", "swap", "return", "exchange",
    "trade",
    "account", "register",
    "stock", "available", "availability",
    "hot", "deals",
    "install", "plan", "program", "subscription",
    "order", "purchase", "buy",
    "address", "change",
}

_FAQ_SECTION_KEYWORDS: dict[str, list[str]] = {
    "Most FAQ": [
        "grade", "grading", "as new", "excellent", "good condition", "fair condition",
        "physical store", "walk in", "pick up", "cancel order", "delivery area",
        "payment method", "hot deals", "split payment",
    ],
    "Product Enquiry": [
        "certified", "second hand", "refurbished", "battery health", "imei",
        "repair service", "stock", "availability", "picture", "photo", "image",
        "original box", "packaging", "source", "country",
    ],
    "Shipping & Services": [
        "delivery status", "track", "tracking", "how long", "shipping time",
        "delayed", "not arrived", "damaged parcel", "wrong item", "shipping number",
        "shipping address", "change address",
    ],
    "Payments": [
        "payment method", "installment", "credit card", "monthly instalment",
        "financing", "pay after", "instalment plan",
    ],
    "Grab PayLater": [
        "paylater", "grab", "grabrewards", "interest free",
    ],
    "SPayLater": [
        "spaylater", "shopee", "late payment", "shopee pay",
    ],
    "ReNewNGo Program": [
        "renewngo", "renew", "36 month", "12 month", "upgrade phone",
        "subscription", "mykad", "monthly charge", "billing", "terminate",
        "outstanding", "payment date", "bank account",
    ],
    "Warranty": [
        "warranty", "defect", "hardware", "claim warranty", "repair",
        "warranty period", "extend warranty", "jailbreak", "root", "sticker",
    ],
    "Replace Plus": [
        "replace plus", "swap", "device swap", "service fee",
    ],
    "Sell Your Device": [
        "sell", "selling", "trade in", "trade-in", "defective device",
        "sell face", "face to face",
    ],
    "My Account": [
        "account", "login", "log in", "password", "register", "newsletter",
        "privacy policy",
    ],
}


# ---------------------------------------------------------------------------
# FAQ document loading
# ---------------------------------------------------------------------------

_faq_sections: dict[str, str] | None = None


def _load_faq_sections() -> dict[str, str]:
    """Parse the FAQ into a dict of {section_name: markdown_text}."""
    global _faq_sections
    if _faq_sections is not None:
        return _faq_sections

    # Prefer the pre-extracted markdown (faster, no dependency on python-docx)
    if FAQ_MD.exists():
        raw = FAQ_MD.read_text(encoding="utf-8")
        sections: dict[str, str] = {}
        current_name = "General"
        current_lines: list[str] = []
        for line in raw.splitlines():
            if line.startswith("## "):
                if current_lines:
                    sections[current_name] = "\n".join(current_lines)
                current_name = line[3:].strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_lines:
            sections[current_name] = "\n".join(current_lines)
        _faq_sections = sections
        return _faq_sections

    # Fallback: parse the docx directly
    try:
        from docx import Document  # type: ignore
    except ImportError:
        _faq_sections = {}
        return _faq_sections

    doc = Document(str(FAQ_DOCX))
    sections = {}
    current_name = "General"
    current_lines: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style.name == "Heading 1":
            if current_lines:
                sections[current_name] = "\n".join(current_lines)
            current_name = text
            current_lines = []
        elif para.style.name == "Heading 2":
            current_lines.append(f"\n### {text}")
        elif para.style.name == "List Paragraph":
            current_lines.append(f"- {text}")
        else:
            current_lines.append(text)

    if current_lines:
        sections[current_name] = "\n".join(current_lines)

    _faq_sections = sections
    return _faq_sections


def _find_relevant_sections(question: str) -> str:
    """Return the most relevant FAQ section text for *question*."""
    q_lower = question.lower()
    sections = _load_faq_sections()

    best_section: str | None = None
    best_score = 0

    for section_name, keywords in _FAQ_SECTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q_lower)
        if score > best_score:
            best_score = score
            best_section = section_name

    if best_section and best_score > 0 and best_section in sections:
        return sections[best_section]

    # No clear winner — return all sections (LLM will pick what's relevant)
    return "\n\n---\n\n".join(sections.values())


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def is_faq_query(text: str) -> bool:
    """
    Return True when the message is asking a CompAsia FAQ question.

    Triggers on:
    - Any single strong FAQ signal (imei, warranty, renewngo, …)
    - Two or more weak signals together (delivery + how long, pay + install, …)
    """
    tokens = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
    # Normalize common compound words
    normalized = text.lower().replace("-", "").replace(" ", "")

    if tokens & _STRONG_FAQ_SIGNALS:
        return True
    if any(sig in normalized for sig in _STRONG_FAQ_SIGNALS):
        return True
    return len(tokens & _WEAK_FAQ_SIGNALS) >= 2


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def build_faq_prompt(question: str, faq_section: str) -> str:
    return (
        "You are a friendly CompAsia customer support assistant. "
        "Answer the customer's question using ONLY the information in the FAQ extract below. "
        "Be concise and conversational. Use bullet points where helpful. "
        "If the FAQ includes a ✅ CHATBOT READY block, include the request for customer details in your reply. "
        "If the answer is not in the FAQ extract, say you don't have that information and "
        "offer: Email support@compasia.com | Call +60 11-6527 3417 | WhatsApp +60 12-941 7355.\n\n"
        f"Customer question: \"{question}\"\n\n"
        f"FAQ extract:\n{faq_section}"
    )


def run_faq_lookup(question: str, provider: str, gemini_client: Any, openai_client: Any,
                   gemini_model: str, openai_model: str) -> dict:
    """
    Find the best FAQ answer for *question* and return a response dict.
    Compatible with the same shape as _run_store_locator().
    """
    faq_section = _find_relevant_sections(question)
    prompt = build_faq_prompt(question, faq_section)

    reply = ""
    try:
        if (provider or "").lower() == "openai":
            response = openai_client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": "Reply as a friendly CompAsia support agent in plain text."},
                    {"role": "user", "content": prompt},
                ],
            )
            reply = response.choices[0].message.content.strip()
        else:
            response = gemini_client.models.generate_content(
                model=gemini_model,
                contents=prompt,
            )
            reply = response.text.strip()
    except Exception as exc:
        print("⚠️ FAQ LLM call failed:", exc)

    if not reply:
        reply = (
            "I'm sorry, I couldn't retrieve the answer right now. "
            "Please contact us at support@compasia.com or call +60 11-6527 3417."
        )

    return {"reply": reply, "source": "faq"}
