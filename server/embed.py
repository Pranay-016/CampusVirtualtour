import hashlib
import json
import os
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer

from env_loader import load_local_env

load_local_env()

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FACULTY_DATA_FILE = os.getenv("FACULTY_CHROMA_JSON", "faculty_chromadb.json")
PLACEMENTS_DATA_FILE = os.getenv("PLACEMENTS_JSON", "placements.json")
COLLEGE_INFO_DATA_FILE = os.getenv("COLLEGE_INFO_JSON", "college_info")

FACULTY_COLLECTION = os.getenv("FACULTY_COLLECTION_NAME", "faculty_collection")
PLACEMENTS_COLLECTION = os.getenv("PLACEMENTS_COLLECTION_NAME", "placements_collection")
COLLEGE_INFO_COLLECTION = os.getenv("COLLEGE_INFO_COLLECTION_NAME", "college_info_collection")


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


def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def make_id(prefix, raw_value):
    digest = hashlib.md5(raw_value.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def titleize(key):
    return key.replace("_", " ").strip().title()


def stringify_value(value):
    if isinstance(value, list):
        return ", ".join(stringify_value(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{titleize(k)}: {stringify_value(v)}" for k, v in value.items())
    return str(value)


def chunked(iterable, size):
    for start in range(0, len(iterable), size):
        yield start, start + size


def load_faculty_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    if not ids or not documents or not metadatas:
        raise ValueError("faculty_chromadb.json must include ids, documents, and metadatas arrays.")

    if not (len(ids) == len(documents) == len(metadatas)):
        raise ValueError("Faculty ids, documents, and metadatas must have the same length.")

    return ids, documents, metadatas


def build_placements_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        rows = json.load(file)

    ids = []
    documents = []
    metadatas = []
    grouped_rows = defaultdict(list)

    for row in rows:
        batch = row.get("academic_batch", "Unknown Batch")
        grouped_rows[batch].append(row)

        year = batch.split("-")[-1] if "-" in batch else batch
        stipend = (
            f"{row['stipend_inr']} INR per month" if row.get("stipend_inr") is not None else "Not specified"
        )
        ctc = f"{row['ctc_lpa']} LPA" if row.get("ctc_lpa") is not None else "Not specified"
        offers = row.get("total_offers", 0)
        sector = row.get("sector") or "Not specified"

        document = (
            f"Placement record for batch {batch} ({year}). "
            f"Company: {row.get('company_name', 'Unknown Company')}. "
            f"CTC: {ctc}. Stipend: {stipend}. "
            f"Sector: {sector}. Total offers: {offers}."
        )

        raw_key = f"{batch}|{row.get('company_name')}|{row.get('ctc_lpa')}|{row.get('stipend_inr')}|{offers}"
        ids.append(make_id("placement", raw_key))
        documents.append(document)
        metadatas.append(
            {
                "doc_type": "company_record",
                "academic_batch": batch,
                "year": year,
                "company_name": str(row.get("company_name", "")),
                "sector": sector,
            }
        )

    for batch, batch_rows in grouped_rows.items():
        year = batch.split("-")[-1] if "-" in batch else batch
        valid_ctcs = [row["ctc_lpa"] for row in batch_rows if row.get("ctc_lpa") is not None]
        highest_ctc = max(valid_ctcs) if valid_ctcs else None
        total_offers = sum(int(row.get("total_offers") or 0) for row in batch_rows)
        recruiter_count = len({row.get("company_name") for row in batch_rows if row.get("company_name")})
        top_recruiters = sorted(
            batch_rows,
            key=lambda row: (int(row.get("total_offers") or 0), float(row.get("ctc_lpa") or 0)),
            reverse=True,
        )[:10]
        top_recruiter_text = ", ".join(
            f"{row.get('company_name')} ({row.get('total_offers', 0)} offers)" for row in top_recruiters
        )

        summary_document = (
            f"Placement summary for batch {batch} corresponding to placement year {year}. "
            f"Highest package: {highest_ctc} LPA. "
            f"Total offers recorded: {total_offers}. "
            f"Unique recruiters: {recruiter_count}. "
            f"Top recruiters by offers: {top_recruiter_text}."
        )

        ids.append(make_id("placement_summary", batch))
        documents.append(summary_document)
        metadatas.append(
            {
                "doc_type": "batch_summary",
                "academic_batch": batch,
                "year": year,
                "company_name": "",
                "sector": "summary",
            }
        )

    overall_top = sorted(
        rows,
        key=lambda row: float(row.get("ctc_lpa") or 0),
        reverse=True,
    )[:10]
    overall_document = (
        "Overall placement highlights across the uploaded datasets for 2023, 2024, and 2025. "
        + "Top packages include: "
        + ", ".join(
            f"{row.get('company_name')} in batch {row.get('academic_batch')} with {row.get('ctc_lpa')} LPA"
            for row in overall_top
        )
        + "."
    )
    ids.append(make_id("placement_overall", "overall"))
    documents.append(overall_document)
    metadatas.append(
        {
            "doc_type": "overall_summary",
            "academic_batch": "all",
            "year": "all",
            "company_name": "",
            "sector": "summary",
        }
    )

    return ids, documents, metadatas


def add_college_info_docs(ids, documents, metadatas, prefix, value, path_parts):
    title_path = " > ".join(titleize(part) for part in path_parts)

    if isinstance(value, dict):
        body = "\n".join(f"{titleize(k)}: {stringify_value(v)}" for k, v in value.items())
        document = f"College information section: {title_path}\n{body}"
        ids.append(make_id(prefix, title_path + body))
        documents.append(document)
        metadatas.append(
            {
                "doc_type": "section_summary",
                "section": title_path,
                "topic": path_parts[0],
            }
        )

        for key, child_value in value.items():
            add_college_info_docs(ids, documents, metadatas, prefix, child_value, path_parts + [key])
        return

    if isinstance(value, list):
        list_text = "\n".join(f"- {stringify_value(item)}" for item in value)
        document = f"College information list: {title_path}\n{list_text}"
        ids.append(make_id(prefix, title_path + list_text))
        documents.append(document)
        metadatas.append(
            {
                "doc_type": "section_list",
                "section": title_path,
                "topic": path_parts[0],
            }
        )
        return

    document = f"College information: {title_path}\nValue: {stringify_value(value)}"
    ids.append(make_id(prefix, title_path + str(value)))
    documents.append(document)
    metadatas.append(
        {
            "doc_type": "section_value",
            "section": title_path,
            "topic": path_parts[0],
        }
    )


def build_college_info_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    ids = []
    documents = []
    metadatas = []

    for key, value in data.items():
        add_college_info_docs(ids, documents, metadatas, "college_info", value, [key])

    return ids, documents, metadatas


def recreate_collection(client, collection_name):
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection '{collection_name}'.")
    except Exception:
        print(f"No existing collection named '{collection_name}' found. Creating a new one.")

    return client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def upload_dataset(client, model, collection_name, ids, documents, metadatas):
    print(f"Preparing collection '{collection_name}' with {len(ids)} documents...")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()
    collection = recreate_collection(client, collection_name)

    batch_size = 128
    for start, end in chunked(ids, batch_size):
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
            documents=documents[start:end],
        )
        print(f"{collection_name}: uploaded {min(end, len(ids))}/{len(ids)}")

    print(f"Successfully stored {len(ids)} records in '{collection_name}'.")


def embed_and_store_all():
    client = get_cloud_client()
    model = get_embedding_model()

    faculty_ids, faculty_docs, faculty_meta = load_faculty_dataset(FACULTY_DATA_FILE)
    placements_ids, placements_docs, placements_meta = build_placements_dataset(PLACEMENTS_DATA_FILE)
    college_ids, college_docs, college_meta = build_college_info_dataset(COLLEGE_INFO_DATA_FILE)

    upload_dataset(client, model, FACULTY_COLLECTION, faculty_ids, faculty_docs, faculty_meta)
    upload_dataset(client, model, PLACEMENTS_COLLECTION, placements_ids, placements_docs, placements_meta)
    upload_dataset(client, model, COLLEGE_INFO_COLLECTION, college_ids, college_docs, college_meta)


if __name__ == "__main__":
    embed_and_store_all()
