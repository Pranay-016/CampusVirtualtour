import json
import os
import re
import sys
from contextlib import redirect_stderr

import chromadb
from sentence_transformers import SentenceTransformer
from env_loader import load_local_env

load_local_env()

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "faculty_collection")
TOP_K = int(os.getenv("CHROMA_TOP_K", "5"))
FETCH_K = int(os.getenv("CHROMA_FETCH_K", "25"))
_MODEL = None
STOPWORDS = {
    "a", "an", "and", "are", "about", "can", "could", "department",
    "do", "for", "from", "give", "head", "help", "hod", "i", "in",
    "is", "me", "of", "please", "tell", "the", "to", "what", "who",
}
DEPARTMENT_PATTERNS = [
    r"(?:hod|head of department)\s+(?:of|for|in)\s+([a-z0-9&/ ,.-]+)",
    r"department\s+(?:of|for|in)\s+([a-z0-9&/ ,.-]+)",
]


def get_embedding_model():
    global _MODEL
    if _MODEL is None:
        with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stderr(devnull):
            _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def get_cloud_client():
    api_key = os.getenv("CHROMA_CLOUD_API_KEY")
    tenant = os.getenv("CHROMA_CLOUD_TENANT")
    database = os.getenv("CHROMA_CLOUD_DATABASE")

    if not api_key or not tenant or not database:
        raise ValueError(
            "Missing Chroma Cloud configuration. Set CHROMA_CLOUD_API_KEY, "
            "CHROMA_CLOUD_TENANT, and CHROMA_CLOUD_DATABASE."
        )

    return chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database,
    )


def normalize_text(text):
    normalized = text.lower()
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("dept.", "department")
    normalized = normalized.replace("dept:", "department ")
    normalized = normalized.replace("dept", "department")
    normalized = normalized.replace("head of department", "hod")
    normalized = normalized.replace("h.o.d", "hod")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize(text):
    return [token for token in normalize_text(text).split() if token and token not in STOPWORDS]


def build_candidate_text(item):
    metadata = item.get("metadata") or {}
    parts = [
        item.get("document", ""),
        metadata.get("name", ""),
        metadata.get("department", ""),
        metadata.get("designation", ""),
        metadata.get("qualification", ""),
    ]
    return " ".join(part for part in parts if part)


def extract_department_phrase(query_text):
    lowered = query_text.lower().strip(" ?.!," )
    for pattern in DEPARTMENT_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            phrase = match.group(1).strip(" ?!.,")
            if phrase:
                return phrase
    return None


def build_department_variants(department_phrase):
    if not department_phrase:
        return []

    cleaned = " ".join(department_phrase.replace("&", " & ").split())
    upper = cleaned.upper()
    title = cleaned.title()

    variants = [
        cleaned,
        upper,
        title,
        f"DEPT: {upper}",
        f"DEPT : {upper}",
        f"Dept: {title}",
        f"Dept : {title}",
        f"Department: {upper}",
        f"Department: {title}",
    ]

    deduped = []
    seen = set()
    for variant in variants:
        key = variant.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(variant)

    return deduped


def merge_candidates(*candidate_lists):
    merged = {}
    for candidate_list in candidate_lists:
        for item in candidate_list:
            key = item.get("id") or item.get("document")
            if not key:
                continue

            existing = merged.get(key)
            if existing is None:
                merged[key] = item
                continue

            existing_distance = existing.get("distance")
            new_distance = item.get("distance")
            if existing_distance is None or (
                isinstance(new_distance, (int, float))
                and isinstance(existing_distance, (int, float))
                and new_distance < existing_distance
            ):
                merged[key] = item

    return list(merged.values())


def rerank_results(query_text, candidates, top_k):
    query_normalized = normalize_text(query_text)
    query_tokens = set(tokenize(query_text))
    department_phrase = extract_department_phrase(query_text)
    department_phrase_normalized = normalize_text(department_phrase) if department_phrase else ""

    def score(item):
        candidate_text = build_candidate_text(item)
        candidate_normalized = normalize_text(candidate_text)
        candidate_tokens = set(tokenize(candidate_text))
        metadata = item.get("metadata") or {}
        designation_normalized = normalize_text(metadata.get("designation", ""))
        department_normalized = normalize_text(metadata.get("department", ""))

        overlap = len(query_tokens & candidate_tokens)
        score_value = overlap * 5

        if "hod" in query_normalized and "hod" in designation_normalized:
            score_value += 20

        if department_normalized and department_normalized in query_normalized:
            score_value += 25

        if department_normalized:
            department_tokens = set(tokenize(department_normalized))
            score_value += len(query_tokens & department_tokens) * 8

        if department_phrase_normalized and department_phrase_normalized in candidate_normalized:
            score_value += 40

        if metadata.get("name") and normalize_text(metadata["name"]) in query_normalized:
            score_value += 30

        distance = item.get("distance")
        if isinstance(distance, (int, float)):
            score_value -= distance * 3

        return score_value

    ranked = sorted(candidates, key=lambda item: score(item), reverse=True)
    return ranked[:top_k]


def vector_search(collection, query_embedding, fetch_k):
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=fetch_k,
    )

    retrieved_data = []
    if results and results.get("documents") and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            distance = results["distances"][0][i] if results.get("distances") else None
            doc_id = results["ids"][0][i] if results.get("ids") else None
            retrieved_data.append(
                {
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta,
                    "distance": distance,
                }
            )
    return retrieved_data


def keyword_search(collection, query_text):
    department_phrase = extract_department_phrase(query_text)
    if not department_phrase:
        return []

    retrieved_data = []
    for department_variant in build_department_variants(department_phrase):
        try:
            results = collection.get(
                where={"department": department_variant},
                limit=50,
                include=["documents", "metadatas"],
            )
        except Exception:
            continue

        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        for i, doc in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc_id = ids[i] if i < len(ids) else None
            retrieved_data.append(
                {
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta,
                    "distance": None,
                }
            )

    return retrieved_data


def query_chroma(query_text, collection_name=COLLECTION_NAME, top_k=TOP_K, fetch_k=FETCH_K):
    try:
        model = get_embedding_model()
        query_embedding = model.encode([query_text]).tolist()

        client = get_cloud_client()
        collection = client.get_collection(name=collection_name)

        vector_candidates = vector_search(collection, query_embedding, fetch_k)
        keyword_candidates = keyword_search(collection, query_text)
        retrieved_data = merge_candidates(vector_candidates, keyword_candidates)
        retrieved_data = rerank_results(query_text, retrieved_data, top_k)
        print(json.dumps({"status": "success", "data": retrieved_data}))
    except Exception as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        collection_name = sys.argv[2] if len(sys.argv) > 2 else COLLECTION_NAME
        query_chroma(query, collection_name=collection_name)
    else:
        print(json.dumps({"status": "error", "message": "No query provided"}))
