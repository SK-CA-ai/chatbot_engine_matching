"""Tests for store_locator.py and the /store-locator Flask endpoint."""
from __future__ import annotations

import io
import sys
import textwrap
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub packages unavailable in the test environment (legacy openai SDK,
# psycopg) so the Flask app can be imported without a real DB or old SDK.
# Must be set before any project modules are imported.
# ---------------------------------------------------------------------------
_openai_stub = MagicMock()
_openai_stub.OpenAI = MagicMock  # only needed for import; Gemini is used for testing
sys.modules["openai"] = _openai_stub
sys.modules["psycopg"] = MagicMock()

from store_locator import (
    build_store_locator_prompt,
    detect_language,
    detect_location,
    find_matching_stores,
    is_location_query,
    load_stores,
)

# ---------------------------------------------------------------------------
# Shared fixture — a small in-memory CSV that mirrors the real schema
# ---------------------------------------------------------------------------

SAMPLE_CSV = textwrap.dedent("""\
    id,name,whatsappLink,location,operatingHours
    1,Store KL Sentral,https://wa.me/111,"Unit 7A, Level 1, Stesen Sentral Kuala Lumpur, 50470 Kuala Lumpur, Wilayah Persekutuan Kuala Lumpur",10AM-10PM
    2,Mydin USJ (Kiosk),https://wa.me/222,"FPL-22, Persiaran Subang Permai, USJ 1, 47500 Subang Jaya, Selangor",10AM-10PM
    3,Angsana Mall JB (Kiosk),https://wa.me/333,"K4.7, Level 4, Angsana Johor Bahru Mall, 81200 Johor Bahru, Johor",10AM-10PM
    4,Closed Outlet Petrajaya (Temporarily Closed),https://wa.me/444,"Mydin Petrajaya, 93050, Sarawak",10AM-10PM
    5,Vivacity Kuching (Kiosk),https://wa.me/555,"Lot L2-K003, Level 2, Vivacity Megamall, 93350 Kuching, Sarawak",10AM-10PM
    6,Mydin Seremban 2 (Kiosk),https://wa.me/666,"Lot GPL-29, Mydin Mall Seremban 2, 70300 Seremban, Negeri Sembilan",10AM-10PM
""")


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(io.StringIO(SAMPLE_CSV))


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_english_message(self):
        assert detect_language("where can I buy near KL?") == "en"

    def test_malay_message(self):
        assert detect_language("kedai dekat penang ada tak?") == "ms"

    def test_malay_with_location(self):
        assert detect_language("ada outlet kat seremban boleh pergi?") == "ms"

    def test_mixed_defaults_to_english(self):
        # Only one Malay word — below the threshold of 2
        assert detect_language("store dekat KL") == "en"

    def test_empty_string(self):
        assert detect_language("") == "en"


# ---------------------------------------------------------------------------
# detect_location
# ---------------------------------------------------------------------------

class TestDetectLocation:
    def test_full_state_name(self):
        assert detect_location("any store in Selangor?") == "selangor"

    def test_abbreviation_kl(self):
        assert detect_location("store near KL") == "kl"

    def test_abbreviation_jb(self):
        assert detect_location("outlet in JB please") == "jb"

    def test_multi_word_beats_single_word(self):
        # "johor bahru" is longer than "johor" — must win
        assert detect_location("nearest store in Johor Bahru") == "johor bahru"

    def test_multi_word_phrase(self):
        assert detect_location("I am in Subang Jaya") == "subang jaya"

    def test_case_insensitive(self):
        assert detect_location("I live in PENANG") == "penang"

    def test_no_location(self):
        assert detect_location("where can I buy?") is None

    def test_partial_word_not_matched(self):
        # "kt" must not match inside "kota" — word boundary required
        assert detect_location("kota bharu store") == "kota bharu"

    def test_malay_sentence(self):
        assert detect_location("kedai kat kuching ada?") == "kuching"

    def test_sarawak(self):
        assert detect_location("outlet in Sarawak") == "sarawak"

    def test_ampang(self):
        assert detect_location("where is the nearest location in ampang") == "ampang"


# ---------------------------------------------------------------------------
# find_matching_stores
# ---------------------------------------------------------------------------

class TestFindMatchingStores:
    def test_finds_active_stores(self, sample_df):
        active, closed = find_matching_stores("kuala lumpur", sample_df)
        names = [s["name"] for s in active]
        assert any("KL Sentral" in n for n in names)
        assert closed == []

    def test_splits_closed_stores(self, sample_df):
        active, closed = find_matching_stores("sarawak", sample_df)
        active_names = [s["name"] for s in active]
        closed_names = [s["name"] for s in closed]
        assert any("Vivacity" in n for n in active_names)
        assert any("Temporarily Closed" in n for n in closed_names)
        # Closed store must NOT appear in active list
        assert not any("Temporarily Closed" in n for n in active_names)

    def test_no_match_returns_empty(self, sample_df):
        active, closed = find_matching_stores("perlis", sample_df)
        assert active == []
        assert closed == []

    def test_state_level_match(self, sample_df):
        active, _ = find_matching_stores("johor", sample_df)
        assert len(active) >= 1
        assert any("JB" in s["name"] or "Johor" in s["location"] for s in active)

    def test_store_dict_has_required_keys(self, sample_df):
        active, _ = find_matching_stores("seremban", sample_df)
        assert len(active) == 1
        store = active[0]
        assert {"name", "location", "whatsappLink", "operatingHours"} <= store.keys()

    def test_alias_kl_matches_kuala_lumpur(self, sample_df):
        from store_locator import LOCATION_MAP
        assert "kl" in LOCATION_MAP
        active, _ = find_matching_stores("kl", sample_df)
        assert len(active) >= 1

    def test_ampang_finds_ampang_point(self):
        df = load_stores()
        active, closed = find_matching_stores("ampang", df)
        names = [s["name"] for s in active]
        assert any("Ampang" in n for n in names), f"Expected Ampang store, got: {names}"
        assert closed == []


# ---------------------------------------------------------------------------
# build_store_locator_prompt
# ---------------------------------------------------------------------------

class TestBuildStoreLocatorPrompt:
    def test_english_prompt_contains_store_name(self, sample_df):
        active, _ = find_matching_stores("kuala lumpur", sample_df)
        prompt = build_store_locator_prompt("store near KL", active, "en")
        assert "KL Sentral" in prompt
        assert "WhatsApp" in prompt

    def test_malay_prompt_in_malay(self, sample_df):
        active, _ = find_matching_stores("seremban", sample_df)
        prompt = build_store_locator_prompt("kedai seremban", active, "ms")
        assert "Melayu" in prompt or "Bahasa" in prompt

    def test_caps_at_five_stores(self):
        many_stores = [
            {"name": f"Store {i}", "location": f"Loc {i}", "whatsappLink": "https://wa.me/x", "operatingHours": "10AM-10PM"}
            for i in range(10)
        ]
        prompt = build_store_locator_prompt("stores near me", many_stores, "en")
        # Only first 5 should appear
        assert "Store 4" in prompt
        assert "Store 5" not in prompt


# ---------------------------------------------------------------------------
# load_stores
# ---------------------------------------------------------------------------

class TestLoadStores:
    def test_loads_real_csv(self):
        import store_locator as sl
        sl._stores_df = None  # reset cache
        df = sl.load_stores()
        assert len(df) > 0
        assert "name" in df.columns
        assert "location" in df.columns
        assert "whatsappLink" in df.columns
        assert "operatingHours" in df.columns

    def test_cache_returns_same_object(self):
        import store_locator as sl
        sl._stores_df = None
        df1 = sl.load_stores()
        df2 = sl.load_stores()
        assert df1 is df2


# ---------------------------------------------------------------------------
# Flask endpoint — /store-locator
# ---------------------------------------------------------------------------

@pytest.fixture
def flask_client():
    """Return a Flask test client with Gemini patched out."""
    import store_locator as sl
    sl._stores_df = None  # reset CSV cache so the real file is re-read

    mock_response = MagicMock()
    mock_response.text = "Here are the stores near you!"

    with patch("engine_matching_flask_api.genai") as mock_genai:
        mock_genai.Client.return_value.models.generate_content.return_value = mock_response

        import engine_matching_flask_api as api
        api.app.config["TESTING"] = True
        with api.app.test_client() as client:
            yield client


class TestStoreLocatorEndpoint:
    def test_missing_user_message_returns_400(self, flask_client):
        r = flask_client.post("/store-locator", json={})
        assert r.status_code == 400
        assert "error" in r.get_json()

    def test_empty_user_message_returns_400(self, flask_client):
        r = flask_client.post("/store-locator", json={"user_message": "  "})
        assert r.status_code == 400

    def test_no_location_returns_needs_location_true(self, flask_client):
        r = flask_client.post("/store-locator", json={"user_message": "where can I buy?"})
        assert r.status_code == 200
        data = r.get_json()
        assert data["needs_location"] is True
        assert data["stores"] == []
        assert len(data["reply"]) > 0

    def test_english_no_location_reply_in_english(self, flask_client):
        r = flask_client.post("/store-locator", json={"user_message": "where can I buy?"})
        data = r.get_json()
        assert "area" in data["reply"].lower() or "city" in data["reply"].lower()

    def test_malay_no_location_reply_in_malay(self, flask_client):
        r = flask_client.post("/store-locator", json={"user_message": "nak beli kat mana boleh?"})
        data = r.get_json()
        assert data["needs_location"] is True
        assert "kawasan" in data["reply"].lower() or "bandar" in data["reply"].lower()

    def test_location_found_returns_stores(self, flask_client):
        r = flask_client.post("/store-locator", json={"user_message": "any store in KL?"})
        assert r.status_code == 200
        data = r.get_json()
        assert data["needs_location"] is False
        assert len(data["stores"]) > 0
        assert data["location_detected"] == "kl"

    def test_response_has_all_keys(self, flask_client):
        r = flask_client.post("/store-locator", json={"user_message": "store in Penang"})
        data = r.get_json()
        for key in ("needs_location", "stores", "closed_stores", "reply", "language", "location_detected"):
            assert key in data, f"missing key: {key}"

    def test_store_objects_have_required_fields(self, flask_client):
        r = flask_client.post("/store-locator", json={"user_message": "store in Penang"})
        data = r.get_json()
        for store in data["stores"]:
            for field in ("name", "location", "whatsappLink", "operatingHours"):
                assert field in store

    def test_closed_stores_separated(self, flask_client):
        r = flask_client.post("/store-locator", json={"user_message": "outlet in Sarawak"})
        data = r.get_json()
        active_names = [s["name"] for s in data["stores"]]
        closed_names = [s["name"] for s in data["closed_stores"]]
        assert not any("Temporarily Closed" in n for n in active_names)
        assert any("Temporarily Closed" in n for n in closed_names)

    def test_ampang_endpoint_returns_store(self, flask_client):
        r = flask_client.post("/store-locator", json={"user_message": "where is the nearest location in ampang"})
        assert r.status_code == 200
        data = r.get_json()
        assert data["needs_location"] is False
        assert data["location_detected"] == "ampang"
        names = [s["name"] for s in data["stores"]]
        assert any("Ampang" in n for n in names), f"Expected Ampang store, got: {names}"

    def test_unrecognised_location_asks_for_clarification(self, flask_client):
        # "Antarctica" is not in LOCATION_MAP → treated the same as no location
        r = flask_client.post("/store-locator", json={"user_message": "store in Antarctica"})
        data = r.get_json()
        assert data["needs_location"] is True
        assert data["stores"] == []
        assert len(data["reply"]) > 0

    def test_llm_failure_falls_back_gracefully(self, flask_client):
        with patch("engine_matching_flask_api.genai") as mock_genai:
            mock_genai.Client.return_value.models.generate_content.side_effect = Exception("LLM down")
            r = flask_client.post("/store-locator", json={"user_message": "store in KL"})
            assert r.status_code == 200
            data = r.get_json()
            assert len(data["reply"]) > 0  # fallback text is present


# ---------------------------------------------------------------------------
# is_location_query
# ---------------------------------------------------------------------------

class TestIsLocationQuery:
    def test_nearest_store(self):
        assert is_location_query("where is the nearest store?") is True

    def test_nearest_outlet(self):
        assert is_location_query("nearest outlet in KL") is True

    def test_branch_question(self):
        assert is_location_query("do you have a branch in Penang?") is True

    def test_kedai_malay(self):
        assert is_location_query("ada kedai kat ampang?") is True

    def test_cawangan_malay(self):
        assert is_location_query("cawangan dekat johor?") is True

    def test_where_and_location(self):
        assert is_location_query("where is the location in ampang") is True

    def test_find_store(self):
        assert is_location_query("where can I find a store near me") is True

    def test_product_question_not_triggered(self):
        assert is_location_query("where can I buy a refurbished phone?") is False

    def test_price_question_not_triggered(self):
        assert is_location_query("what is the price of iPhone 13?") is False

    def test_empty_string(self):
        assert is_location_query("") is False


# ---------------------------------------------------------------------------
# /engine-match intercept — location queries routed to store locator
# ---------------------------------------------------------------------------

class TestEngineMatchStoreLocatorIntercept:
    def test_location_question_returns_store_locator_match(self, flask_client):
        r = flask_client.post("/engine-match", json={"question": "where is the nearest outlet in ampang"})
        assert r.status_code == 200
        data = r.get_json()
        assert data["match"] == "STORE_LOCATOR"
        assert data["score"] == 1.0
        assert data["matched_row"] is None
        assert "store_locator" in data

    def test_location_question_store_locator_has_stores(self, flask_client):
        r = flask_client.post("/engine-match", json={"question": "nearest store in KL"})
        data = r.get_json()
        assert data["match"] == "STORE_LOCATOR"
        sl = data["store_locator"]
        assert sl["needs_location"] is False
        assert len(sl["stores"]) > 0
        assert sl["location_detected"] == "kl"

    def test_no_location_in_question_asks_for_it(self, flask_client):
        r = flask_client.post("/engine-match", json={"question": "where is the nearest outlet?"})
        data = r.get_json()
        assert data["match"] == "STORE_LOCATOR_NEEDS_LOCATION"
        assert data["store_locator"]["needs_location"] is True

    def test_malay_location_question_intercepted(self, flask_client):
        r = flask_client.post("/engine-match", json={"question": "ada kedai kat penang?"})
        data = r.get_json()
        assert data["match"] == "STORE_LOCATOR"
        assert data["store_locator"]["language"] == "ms"

    def test_non_location_question_not_intercepted(self, flask_client):
        # product question — should NOT be intercepted by store locator
        # engine_match will raise or return something, but match must NOT be STORE_LOCATOR
        with patch("engine_matching_flask_api.engine_match") as mock_em:
            mock_em.return_value = ("PRODUCT_ENQUIRE", 0.9, [])
            r = flask_client.post("/engine-match", json={"question": "how much is iPhone 13?"})
            data = r.get_json()
            assert data["match"] != "STORE_LOCATOR"
            assert data["match"] == "PRODUCT_ENQUIRE"
