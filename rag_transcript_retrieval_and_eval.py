# -*- coding: utf-8 -*-
"""
RAG and Eval - Reference-Based Only (No QAFactEval) [Recursive Grounded Summarization]
"""

# --- 📚 Imports ---
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import numpy as np
import pandas as pd
import pickle
import faiss
import torch

# ✅ Load persisted variables
with open("data/gold_labels.pkl", "rb") as f:
    gold_labels = pickle.load(f)

with open("data/justia_chunks.pkl", "rb") as f:
    justia_chunks = pickle.load(f)

with open("data/convos_chunks.pkl", "rb") as f:
    convos_chunks = pickle.load(f)

with open("data/indexed_docs.pkl", "rb") as f:
    indexed_docs = pickle.load(f)

# ✅ Load embeddings
embeddings_np = np.load("data/embeddings.npy")

# ✅ Load FAISS index
index = faiss.read_index("data/justia_convos.index")

# ✅ Load LexT5
tokenizer = AutoTokenizer.from_pretrained("santoshtyss/lt5-large", use_auth_token=True)
model = AutoModelForSeq2SeqLM.from_pretrained("santoshtyss/lt5-large", use_auth_token=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ✅ Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# ✅ Retrieval function
TOP_K = 3

def retrieve_justia_chunks(query_vector, case_id, top_k=TOP_K):
    distances, indices = index.search(query_vector, 50)
    justia_indices = [
        idx for idx in indices[0]
        if idx < len(justia_chunks)
        and indexed_docs[idx].metadata.get("case_id") == case_id
    ]
    justia_indices = justia_indices[:top_k]
    return [indexed_docs[idx] for idx in justia_indices]

# ✅ Summarize a single transcript chunk

def summarize_chunk(chunk_text):
    prompt = f"""
Below is a part of a legal document:
--
{chunk_text}
--
We are creating one comprehensive summary for the legal document by recursively merging summaries of its chunks. Now, write a summary for the excerpt provided above, making sure to include vital information related to legal arguments, backgrounds, legal settings, key figures, their objectives, and motivations. If a legal norm or code is cited, it must be correct and include the right number. Summarize all key events and everything that is relevant to the case. Be concise and use legal notation and language. The summary must be within 100 words and could include multiple paragraphs.
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ Summarize all transcript chunks for a case

def summarize_chunks_group(convo_chunks_for_case):
    chunk_summaries = []
    for doc in convo_chunks_for_case:
        chunk_summary = summarize_chunk(doc.page_content)
        chunk_summaries.append(chunk_summary)
    return " ".join(chunk_summaries)

# ✅ Generate final grounded summary using Justia

def final_grounded_summary(intermediate_summary, support_docs, case_id):
    retrieved_opinion_chunks_text = "\n".join(
        [f"[{j+1}] {doc.page_content}" for j, doc in enumerate(support_docs)]
    )
    final_prompt = f"""
Below is a summary of the context preceding some parts of a legal document:
--
{intermediate_summary}
--

Below are several summaries of consecutive parts of a legal document:
--
{retrieved_opinion_chunks_text}
--
We are merging the preceding context and the summaries into one comprehensive summary. This summary should include who is involved, when it happened, to whom it concerns, on what legal basis, and a location reference. Ensure to incorporate vital information related to legal arguments, backgrounds, legal settings, key figures, their objectives, and motivations. Despite the complexity, the summary must present a coherent argument in one concise form.
"""
    inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ Main pipeline
results = []
case_ids = sorted({doc.metadata["case_id"] for doc in convos_chunks})
case_ids = [cid for cid in case_ids if cid in gold_labels]
case_ids = case_ids[:5]  # Limit for test run

print(f"[INFO] Running on {len(case_ids)} cases with gold labels")

for case_id in case_ids:
    convo_docs = [doc for doc in convos_chunks if doc.metadata["case_id"] == case_id]
    convos_idxs = [i for i, doc in enumerate(indexed_docs) if doc.metadata.get("case_id") == case_id and i >= len(justia_chunks)]
    if not convo_docs or not convos_idxs:
        continue

    intermediate_summary = summarize_chunks_group(convo_docs)
    query_vector = np.mean(embeddings_np[convos_idxs], axis=0).astype('float32').reshape(1, -1)
    support_docs = retrieve_justia_chunks(query_vector, case_id)
    if not support_docs:
        continue

    final_summary = final_grounded_summary(intermediate_summary, support_docs, case_id)

    if case_id in gold_labels:
        ref = gold_labels[case_id]
        rouge_score = scorer.score(ref, final_summary)
        print(f"[ROUGE] for Case {case_id}: {rouge_score}")
    else:
        rouge_score = None

    print(f"\n📌 [SUMMARY] Case {case_id}:\n{final_summary}\n")

    results.append({
        "case_id": case_id,
        "summary": final_summary,
        "rouge_score": rouge_score
    })

# ✅ Save results
pd.DataFrame(results).to_csv("data/summaries_and_scores_TEST.csv", index=False)
print("\n✅ Recursive summarization results saved to data/summaries_and_scores_TEST.csv")
