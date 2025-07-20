import os
import torch
import pickle
import numpy as np
import pandas as pd
import faiss
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer

# === Setup Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Constants ===
MAX_CHUNK_INPUT_TOKENS = 512
MAX_CHUNK_SUMMARY_TOKENS = 384
MAX_FINAL_INPUT_TOKENS = 4096
MAX_FINAL_SUMMARY_TOKENS = 512
MIN_FINAL_SUMMARY_TOKENS = 64
TOP_K = 3

# === Load Persisted Files ===
with open("data/gold_labels.pkl", "rb") as f:
    gold_labels = pickle.load(f)
with open("data/justia_chunks.pkl", "rb") as f:
    justia_chunks = pickle.load(f)
with open("data/convos_chunks.pkl", "rb") as f:
    convos_chunks = pickle.load(f)
with open("data/indexed_docs.pkl", "rb") as f:
    indexed_docs = pickle.load(f)
embeddings_np = np.load("data/embeddings.npy")
index = faiss.read_index("data/justia_convos.index")

# === Load Model and Tokenizer ===
model_name = "santoshtyss/lt5-longlarge"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === Load Embedding Model ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === ROUGE Setup ===
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# === Main Pipeline ===
results = []
case_ids = sorted({doc.metadata["case_id"] for doc in convos_chunks if doc.metadata["case_id"] in gold_labels})

for case_id in case_ids[:5]:
    transcript_chunks = [doc.page_content for doc in convos_chunks if doc.metadata["case_id"] == case_id]
    if not transcript_chunks:
        continue

    # Stage 1: Summarize transcript chunks
    intermediate_summaries = []
    for i, chunk_text in enumerate(transcript_chunks, start=1):
        prompt = (
                     "You are a legal assistant. Summarize this excerpt from a U.S. Supreme Court oral argument, "
                     "highlighting the key legal issues, arguments, and questions presented:\n\n"
                     f"{chunk_text}"
                    )
        tokens = tokenizer(prompt, max_length=MAX_CHUNK_INPUT_TOKENS, truncation=True, return_tensors="pt").to(device)
        input_ids = tokens["input_ids"]
        orig_token_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if orig_token_len > MAX_CHUNK_INPUT_TOKENS:
            logger.warning(f"Chunk {i} prompt was {orig_token_len} tokens; truncated.")
        else:
            logger.info(f"Chunk {i} prompt length: {orig_token_len} tokens.")
        summary_ids = model.generate(**tokens, max_new_tokens=MAX_CHUNK_SUMMARY_TOKENS, num_beams=4, early_stopping=True)
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        intermediate_summaries.append(summary_text)
        with open("debug_logs.txt", "a", encoding="utf-8") as dbg:
            dbg.write(f"\n--- Chunk {i} ---\nPrompt:\n{prompt[:500]}...\nSummary:\n{summary_text}\n")

    combined_summary = " ".join(intermediate_summaries)
    logger.info(f"Combined intermediate summary length: {len(combined_summary)} chars")

    # Stage 2: Retrieve judgment chunks
    convos_idxs = [i for i, doc in enumerate(indexed_docs) if doc.metadata.get("case_id") == case_id and i >= len(justia_chunks)]
    if not convos_idxs:
        continue
    query_vector = np.mean(embeddings_np[convos_idxs], axis=0).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, 50)
    justia_indices = [
        idx for idx in indices[0]
        if idx < len(justia_chunks) and indexed_docs[idx].metadata.get("case_id") == case_id
    ][:TOP_K]
    top_chunks = [indexed_docs[idx].page_content for idx in justia_indices]
    retrieved_passages_text = "\n".join(top_chunks)

    with open("debug_logs.txt", "a", encoding="utf-8") as dbg:
        dbg.write("\n--- Retrieved Judgment Passages ---\n")
        for j, passage in enumerate(top_chunks, start=1):
            dbg.write(f"Passage {j}:\n{passage}\n\n")

    final_prompt = (
        f"Hearing Summary:\n{combined_summary}\n\n"
        f"Judgment Excerpts:\n{retrieved_passages_text}\n\n"
        "Write a comprehensive summary of the hearing based on the transcript and legal judgment excerpts. "
        "Ensure legal findings are integrated to provide factual support and contextual accuracy."
    )

    final_tokens = tokenizer(final_prompt, max_length=MAX_FINAL_INPUT_TOKENS, truncation=True, return_tensors="pt").to(device)
    orig_final_len = len(tokenizer(final_prompt, add_special_tokens=False)["input_ids"])
    if orig_final_len > MAX_FINAL_INPUT_TOKENS:
        logger.warning(f"Final summary prompt was {orig_final_len} tokens; truncated.")
    else:
        logger.info(f"Final prompt length: {orig_final_len} tokens.")

    final_summary_ids = model.generate(
        **final_tokens,
        max_new_tokens=MAX_FINAL_SUMMARY_TOKENS,
        min_new_tokens=MIN_FINAL_SUMMARY_TOKENS,
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    final_summary = tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)
    print(f"FINAL SUMMARY for {case_id}:\n", final_summary)

    with open("debug_logs.txt", "a", encoding="utf-8") as dbg:
        dbg.write("\n--- Final Prompt ---\n")
        dbg.write(final_prompt[:1000] + ("..." if len(final_prompt) > 1000 else "") + "\n")
        dbg.write("\n=== Final Summary ===\n")
        dbg.write(final_summary + "\n")

    rouge_score = scorer.score(gold_labels[case_id], final_summary)
    results.append({"case_id": case_id, "summary": final_summary, "rouge_score": rouge_score})
    logger.info(f"[ROUGE] for {case_id}: {rouge_score}")

# === Save Output ===
os.makedirs("data", exist_ok=True)
pd.DataFrame(results).to_csv("data/summaries_and_scores_LONG.csv", index=False)
logger.info("✅ Summaries saved to data/summaries_and_scores_LONG.csv")