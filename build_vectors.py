#!/usr/bin/env python3
import os
import json
import hashlib
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import psycopg2
import psycopg2.extras
import faiss
from sentence_transformers import SentenceTransformer
import pyarrow as pa
import pyarrow.parquet as pq


DB_ENV_PATH = "db.env"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_BATCH = 32
BUILD_LIMIT: Optional[int] = None
CACHE_DIR = ".cache_semantic_search"
CACHE_META = "meta.json"
CACHE_INDEX = "index.faiss"
CACHE_EMBEDDINGS = "embeddings.parquet"


def load_env_file(path: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
            elif ":" in line:
                key, val = line.split(":", 1)
            else:
                continue
            env[key.strip()] = val.strip()
    return env


def get_db_conn(env: Dict[str, str]):
    return psycopg2.connect(
        host=env.get("DB_HOST"),
        port=env.get("DB_PORT"),
        dbname=env.get("DB_NAME"),
        user=env.get("DB_USER"),
        password=env.get("DB_PASSWORD"),
    )


def fetch_rows(conn, limit: Optional[int]) -> List[Dict[str, object]]:
    sql = """
        SELECT
            product_id, variant_id, color, spec, condition, price,
            handle, vendor, product_type, tenure
        FROM shopify_variant_new
    """
    if limit is not None:
        sql += " LIMIT %s"
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if limit is not None:
            cur.execute(sql, (limit,))
        else:
            cur.execute(sql)
        return list(cur.fetchall())


def build_sentence(row: Dict[str, object]) -> str:
    def norm(v: object) -> str:
        if v is None:
            return "unknown"
        return str(v).strip()

    vendor = norm(row.get("vendor"))
    product_type = norm(row.get("product_type"))
    handle = norm(row.get("handle"))
    color = norm(row.get("color"))
    spec = norm(row.get("spec"))
    condition = norm(row.get("condition"))
    tenure = norm(row.get("tenure"))
    price = norm(row.get("price"))
    return (
        f"Vendor {vendor} offers a {product_type} with handle {handle}, "
        f"color {color}, spec {spec}, condition {condition}, tenure {tenure}, "
        f"priced at {price}."
    )


def compute_fingerprint(rows: List[Dict[str, object]]) -> str:
    h = hashlib.sha256()
    for r in rows:
        parts = [
            str(r.get("product_id")),
            str(r.get("variant_id")),
            str(r.get("vendor")),
            str(r.get("product_type")),
            str(r.get("handle")),
            str(r.get("color")),
            str(r.get("spec")),
            str(r.get("condition")),
            str(r.get("tenure")),
            str(r.get("price")),
        ]
        h.update("|".join(parts).encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    arr = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return arr.astype(np.float32, copy=False)


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_cache(
    meta: Dict[str, object],
    vectors: np.ndarray,
    id_map: List[Tuple[int, int]],
    index: faiss.IndexFlatIP,
) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    meta_path = os.path.join(CACHE_DIR, CACHE_META)
    index_path = os.path.join(CACHE_DIR, CACHE_INDEX)
    embed_path = os.path.join(CACHE_DIR, CACHE_EMBEDDINGS)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    product_ids = [p for p, _ in id_map]
    variant_ids = [v for _, v in id_map]
    table = pa.table(
        {
            "product_id": pa.array(product_ids, type=pa.int64()),
            "variant_id": pa.array(variant_ids, type=pa.int64()),
            "embedding": pa.array(vectors.tolist(), type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, embed_path)
    faiss.write_index(index, index_path)


def main() -> int:
    if not os.path.exists(DB_ENV_PATH):
        print(f"Missing {DB_ENV_PATH}.")
        return 1

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    env = load_env_file(DB_ENV_PATH)
    start_time = time.time()

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    with get_db_conn(env) as conn:
        print("Fetching rows from database...")
        rows = fetch_rows(conn, BUILD_LIMIT)
        if not rows:
            print("No rows found.")
            return 0

    id_map = [(int(r["product_id"]), int(r["variant_id"])) for r in rows]
    fingerprint = compute_fingerprint(rows)

    print(f"Embedding {len(rows)} rows...")
    texts = [build_sentence(r) for r in rows]
    print(texts)
    vectors = embed_texts(model, texts)
    print("Embedding complete.")
    print("Building FAISS index...")
    index = build_faiss_index(vectors)
    meta = {
        "fingerprint": fingerprint,
        "model": EMBED_MODEL,
        "row_count": len(rows),
        "limit": BUILD_LIMIT,
    }
    print("Saving cache to disk...")
    save_cache(meta, vectors, id_map, index)
    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
