#!/usr/bin/env python3
import os
import re
from typing import Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer
from google import genai

from semantic_search import (
    DB_ENV_PATH,
    EMBED_MODEL,
    GEMINI_MODEL,
    build_search_query,
    fetch_full_records,
    get_db_conn,
    load_cache,
    load_env_file,
    search_index,
)

TOP_K = 3
CANDIDATE_K = 50
SHOP_DOMAIN = os.getenv("SHOP_DOMAIN", "compasia-malaysia.myshopify.com")


def _normalize_domain(shop_domain: str) -> str:
    if not shop_domain:
        return ""
    return shop_domain.replace("https://", "").replace("http://", "").strip("/")


def _build_product_link(handle: str) -> str:
    handle = handle or ""
    domain = _normalize_domain(SHOP_DOMAIN)
    return f"https://{domain}/products/{handle}" if domain else ""


def _build_variant_link(handle: str, variant_id: object) -> str:
    product_link = _build_product_link(handle)
    if not product_link or not variant_id:
        return ""
    return f"{product_link}?variant={variant_id}"


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip().replace(",", "")
    if not raw:
        return None
    raw = raw.replace("RM", "").replace("rm", "").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def _passes_price_filter(price: object, price_min: Optional[float], price_max: Optional[float]) -> bool:
    value = _to_float(price)
    if value is None:
        return True
    if price_min is not None and value < price_min:
        return False
    if price_max is not None and value >= price_max:
        return False
    return True


def _normalize_tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 1]


def _meaningful_model_tokens(text: str) -> List[str]:
    # Remove query-noise terms so availability matching focuses on model identity.
    stopwords = {
        "do",
        "you",
        "guys",
        "have",
        "has",
        "any",
        "in",
        "stock",
        "stocks",
        "available",
        "availability",
        "currently",
        "now",
        "please",
        "can",
        "could",
        "for",
        "with",
        "want",
        "need",
        "looking",
        "series",
    }
    tokens = _normalize_tokens(text)
    cleaned = [t for t in tokens if t not in stopwords]
    # De-duplicate while preserving order.
    seen = set()
    deduped: List[str] = []
    for token in cleaned:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _is_handle_match_for_model(model_tokens: List[str], handle: str) -> bool:
    handle_tokens = set(_normalize_tokens(handle))
    if not handle_tokens or not model_tokens:
        return False

    # Never treat different numeric model generations as matches (e.g. 12 vs 14).
    model_numeric_tokens = [token for token in model_tokens if token.isdigit()]
    if model_numeric_tokens and any(token not in handle_tokens for token in model_numeric_tokens):
        return False

    # For short model names, require exact token presence to avoid contradictions.
    if len(model_tokens) <= 3:
        return all(token in handle_tokens for token in model_tokens)

    required_hits = max(2, (len(model_tokens) * 3 + 3) // 4)  # ~75% rounded up
    hits = sum(1 for token in model_tokens if token in handle_tokens)
    return hits >= required_hits


def _recommended_in_results(recommended_model: str, handles: List[str]) -> bool:
    if not recommended_model or not handles:
        return False
    model_tokens = _meaningful_model_tokens(recommended_model)
    if not model_tokens:
        return False

    for handle in handles:
        if _is_handle_match_for_model(model_tokens, str(handle or "").lower()):
            return True
    return False


def _filter_rows_to_recommended_model(
    rows: List[Dict[str, object]],
    recommended_model: str,
) -> List[Dict[str, object]]:
    if not rows or not recommended_model:
        return rows
    model_tokens = _meaningful_model_tokens(recommended_model)
    if not model_tokens:
        return rows

    filtered = [
        row
        for row in rows
        if _is_handle_match_for_model(model_tokens, str(row.get("handle", "") or "").lower())
    ]
    return filtered or rows


def _filter_rows_to_models_mentioned_in_reply(
    rows: List[Dict[str, object]],
    reply_text: str,
) -> List[Dict[str, object]]:
    if not rows or not reply_text:
        return rows

    normalized_reply = re.sub(r"[^a-z0-9]+", " ", reply_text.lower()).strip()
    if not normalized_reply:
        return rows

    mentioned: List[Dict[str, object]] = []
    for row in rows:
        handle = str(row.get("handle", "") or "").strip().lower()
        if not handle:
            continue
        handle_phrase = re.sub(r"[^a-z0-9]+", " ", handle.replace("-", " ")).strip()
        if handle_phrase and handle_phrase in normalized_reply:
            mentioned.append(row)

    if mentioned:
        return mentioned
    return rows


def _format_cards(rows: List[Dict[str, object]]) -> str:
    def _pretty_handle(handle: object) -> str:
        raw = str(handle or "").strip()
        if not raw:
            return "Unknown model"
        raw = re.sub(r"-flash-deal$", "", raw)
        return " ".join(part.upper() if part.isdigit() else part.capitalize() for part in raw.split("-"))

    def _fmt_field(value: object) -> str:
        text = str(value or "").strip()
        return text if text and text.lower() != "none" else "-"

    def _format_from_price_rm(value: object) -> str:
        amount = _to_float(value)
        if amount is None:
            return "-"
        if float(amount).is_integer():
            return f"RM{int(amount):,}"
        return f"RM{amount:,.2f}"

    def _compact_values(values: object) -> str:
        if isinstance(values, list):
            cleaned = []
            seen = set()
            for v in values:
                text = _fmt_field(v)
                if text == "-" or text in seen:
                    continue
                seen.add(text)
                cleaned.append(text)
            if not cleaned:
                return "-"
            return " / ".join(cleaned)
        return _fmt_field(values)

    blocks: List[str] = []
    for idx, r in enumerate(rows, start=1):
        product_link = str(r.get("product_link", "") or "").strip() or "-"
        options = _compact_values(r.get("storage_options", r.get("spec", "-")))
        conditions = _compact_values(r.get("condition_options", r.get("condition", "-")))
        colors = _compact_values(r.get("color_options", r.get("color", "-")))
        block = (
            f"{idx}. {_pretty_handle(r.get('handle', ''))}\n"
            f"From {_format_from_price_rm(r.get('from_price', r.get('price', '')))}\n"
            f"Options: {options}\n"
            f"Condition: {conditions}\n"
            f"Color: {colors}\n"
            f"View options: {product_link}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def _storage_sort_key(value: object) -> Tuple[int, str]:
    text = str(value or "").strip().lower()
    if not text:
        return (10**9, "")
    match = re.search(r"(\d+)\s*(tb|gb)?", text)
    if not match:
        return (10**9, text)
    qty = int(match.group(1))
    unit = match.group(2) or "gb"
    gb = qty * 1024 if unit == "tb" else qty
    return (gb, text)


def build_diverse_model_rows(
    hits: List[Tuple[int, Tuple[int, int], float]],
    record_map: Dict[Tuple[int, int], Dict[str, object]],
    top_k: int,
    preferred_handles: Optional[set[str]] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    order: List[str] = []

    for rank, key, score in hits:
        rec = record_map.get(key, {})
        handle = str(rec.get("handle", "") or "").strip()
        if not handle:
            continue
        if not _passes_price_filter(rec.get("price"), price_min, price_max):
            continue

        if handle not in grouped:
            grouped[handle] = {
                "first_rank": rank,
                "best_score": float(score),
                "best_key": key,
                "best_price": _to_float(rec.get("price")),
                "min_price": _to_float(rec.get("price")),
                "storage_options": [],
                "condition_options": [],
                "color_options": [],
            }
            order.append(handle)
        group = grouped[handle]

        score_value = float(score)
        price_value = _to_float(rec.get("price"))
        current_best_score = float(group["best_score"])
        current_best_price = group.get("best_price")
        better_by_score = score_value > current_best_score
        better_by_price = (
            score_value == current_best_score
            and price_value is not None
            and (current_best_price is None or price_value < current_best_price)
        )
        if better_by_score or better_by_price:
            group["best_score"] = score_value
            group["best_key"] = key
            group["best_price"] = price_value

        min_price = group.get("min_price")
        if price_value is not None and (min_price is None or price_value < min_price):
            group["min_price"] = price_value

        spec = str(rec.get("spec", "") or "").strip()
        if spec and spec not in group["storage_options"]:
            group["storage_options"].append(spec)
        condition = str(rec.get("condition", "") or "").strip()
        if condition and condition not in group["condition_options"]:
            group["condition_options"].append(condition)
        color = str(rec.get("color", "") or "").strip()
        if color and color not in group["color_options"]:
            group["color_options"].append(color)

    if not grouped:
        return []

    preferred_handles = preferred_handles or set()
    preferred = [h for h in order if h in preferred_handles]
    non_preferred = [h for h in order if h not in preferred_handles]
    selected_handles = (preferred + non_preferred)[:top_k]

    rows: List[Dict[str, object]] = []
    for handle in selected_handles:
        group = grouped[handle]
        best_key = group["best_key"]
        best_rec = record_map.get(best_key, {})
        storage_options = sorted(group["storage_options"], key=_storage_sort_key)
        color_options = group["color_options"]
        rows.append(
            {
                "rank": group["first_rank"],
                "handle": handle,
                "color": best_rec.get("color", ""),
                "spec": best_rec.get("spec", ""),
                "condition": best_rec.get("condition", ""),
                "tenure": best_rec.get("tenure", ""),
                "price": best_rec.get("price", ""),
                "from_price": group["min_price"] if group["min_price"] is not None else best_rec.get("price", ""),
                "storage_options": storage_options,
                "condition_options": group["condition_options"],
                "color_options": color_options,
                "variant_link": _build_variant_link(handle, best_key[1]),
                "product_link": _build_product_link(handle),
            }
        )
    return rows


def _is_refinement_query(text: str) -> bool:
    t = text.lower()
    tokens = _normalize_tokens(t)
    if not tokens:
        return False
    if any(k in t for k in ["gb", "storage", "ram", "under", "below", "above", "between", "range", "budget", "rm", "price"]):
        return True
    return len(tokens) <= 4


def _build_human_reply(
    query: str,
    recommended_model: str,
    recommended_in_results: bool,
    top_rows: List[Dict[str, object]],
    memory: List[Dict[str, str]],
    price_min: Optional[float],
    price_max: Optional[float],
) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    # Keep memory light: last 3 turns
    history = memory[-3:]
    history_text = "\n".join(
        f"User: {m['user']}\nBot: {m['bot']}" for m in history if m.get("user") and m.get("bot")
    )

    catalog_snippet = "\n".join(
        f"- {r['handle']} | {r['color']} | {r['spec']} | {r['condition']} | {r['tenure']} | {r['price']}"
        for r in top_rows
    )

    effective_recommended_model = (
        (recommended_model or "").strip() if recommended_in_results else ""
    )

    system_prompt = (
        "You are a friendly phone recommendation assistant. "
        "Write a short intro reply in 1-2 sentences. "
        "Be concise, natural, and mobile-friendly. "
        "If recommended_model is provided and recommended_in_results is true, mention that matching options are in current stock. "
        "If recommended_in_results is false, do NOT mention any unavailable model name; just offer alternatives from Top results. "
        "If the user insists on unavailable brands or models, ask what other brand or model they would consider. "
        "Briefly explain why the alternatives fit the user's preferences. "
        "If recommended_model is not provided, do not claim a requested model is out of stock. "
        "If recommended_in_results is true, state that matching options are available in current stock. "
        "Do not claim stock or availability beyond what is stated. "
        "Avoid hardcoding model names beyond the provided data. "
        "Do not include bullets, markdown, or product details in this intro."
        "Do not recommend samsung"
    )

    price_line = "None"
    if price_min is not None and price_max is not None:
        price_line = f"{price_min} to {price_max}"
    elif price_min is not None:
        price_line = f"min {price_min}"
    elif price_max is not None:
        price_line = f"max {price_max}"

    prompt = (
        f"{system_prompt}\n\n"
        f"Conversation history (if any):\n{history_text or 'None'}\n\n"
        f"User query: {query}\n"
        f"recommended_model: {effective_recommended_model or 'None'}\n"
        f"recommended_in_results: {recommended_in_results}\n"
        f"price_filter: {price_line}\n"
        f"Top results (use as alternatives):\n{catalog_snippet or 'None'}"
    )

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = (response.text or "").strip()
    return text


def main() -> int:
    if not os.path.exists(DB_ENV_PATH):
        print(f"Missing {DB_ENV_PATH}.")
        return 1

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    env = load_env_file(DB_ENV_PATH)

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    cache = load_cache()
    meta = cache.get("meta", {})
    if not meta:
        print("Cache missing. Run build_vectors.py first.")
        return 1
    if meta.get("model") != EMBED_MODEL:
        print("Cache model mismatch. Rebuild vectors with build_vectors.py.")
        return 1

    index = cache["index"]
    id_map = cache["id_map"]

    memory: List[Dict[str, str]] = []
    last_effective_query: Optional[str] = None
    last_handles: List[str] = []

    while True:
        query = input("Enter search query (or 'exit'): ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            break

        search_query, recommended_model, price_min, price_max = build_search_query(query, None)
        if recommended_model:
            print(f"Gemini recommended model: {recommended_model}")
        if price_min is not None:
            print(f"Gemini parsed price_min: {price_min}")
        if price_max is not None:
            print(f"Gemini parsed price_max: {price_max}")

        effective_query = f"{recommended_model} {search_query}".strip() if recommended_model else search_query
        if _is_refinement_query(query) and last_effective_query and not recommended_model:
            effective_query = f"{last_effective_query} {search_query}".strip()
        if not effective_query and last_effective_query:
            effective_query = last_effective_query
        if effective_query:
            last_effective_query = effective_query
        print(f"Embedding query and searching: {effective_query}")

        scores, idx, q_vec = search_index(model, index, effective_query, CANDIDATE_K)
        # Intentionally suppress low-level debug details for a cleaner UX.

        hits: List[Tuple[int, Tuple[int, int], float]] = []
        for rank, i in enumerate(idx):
            if i < 0 or i >= len(id_map):
                continue
            hits.append((rank + 1, id_map[i], float(scores[rank])))

        if not hits:
            print("No matches.")
            continue

        with get_db_conn(env) as conn:
            keys = [h[1] for h in hits]
            records = fetch_full_records(conn, keys)
            record_map = {(int(r["product_id"]), int(r["variant_id"])): r for r in records}

        preferred_handles = set(last_handles) if _is_refinement_query(query) and last_handles else set()
        rows = build_diverse_model_rows(
            hits=hits,
            record_map=record_map,
            top_k=TOP_K,
            preferred_handles=preferred_handles,
            price_min=price_min,
            price_max=price_max,
        )
        if not rows:
            if price_min is not None and price_max is not None:
                print(f"No matches in range {price_min} to {price_max}.")
            elif price_min is not None:
                print(f"No matches above {price_min}.")
            elif price_max is not None:
                print(f"No matches under {price_max}.")
            else:
                print("No matches after filtering.")
            continue

        handles = [r["handle"] for r in rows if r.get("handle")]
        in_results = _recommended_in_results(recommended_model or "", handles)
        if in_results and recommended_model:
            rows = _filter_rows_to_recommended_model(rows, recommended_model)
            handles = [r["handle"] for r in rows if r.get("handle")]
            in_results = _recommended_in_results(recommended_model or "", handles)

        bot_reply = _build_human_reply(
            query=query,
            recommended_model=recommended_model,
            recommended_in_results=in_results,
            top_rows=rows,
            memory=memory,
            price_min=price_min,
            price_max=price_max,
        )
        rows = _filter_rows_to_models_mentioned_in_reply(rows, bot_reply)

        print("\n" + bot_reply + "\n")
        print(_format_cards(rows))

        memory.append({"user": query, "bot": bot_reply})
        last_handles = [r["handle"] for r in rows if r.get("handle")]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
