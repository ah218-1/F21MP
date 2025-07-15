# -*- coding: utf-8 -*-
"""
RAG and Eval - Reference-Based Only (No QAFactEval) [SMALL TEST VERSION]
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
tokenizer = AutoTokenizer.from_pretrained(
    "santoshtyss/lt5-large", use_auth_token=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "santoshtyss/lt5-large", use_auth_token=True
)
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

# ✅ Main pipeline
results = []
case_ids = sorted({doc.metadata["case_id"] for doc in convos_chunks})
case_ids = [cid for cid in case_ids if cid in gold_labels]

# ⚡️ Run only the first 20 cases
#case_ids = case_ids[:20]

print(f"[INFO] Running on {len(case_ids)} cases with gold labels")

for case_id in case_ids:
    transcripts_text = " ".join(
        [doc.page_content for doc in convos_chunks if doc.metadata.get("case_id") == case_id]
    )
    if not transcripts_text:
        print(f"[WARN] No transcripts found for case {case_id}")
        continue

    justia_text = " ".join(
        [doc.page_content for doc in justia_chunks if doc.metadata.get("case_id") == case_id]
    )

    convos_idxs = [
        i for i, doc in enumerate(indexed_docs)
        if doc.metadata.get("case_id") == case_id and i >= len(justia_chunks)
    ]

    print(f"[INFO] Case {case_id} — convos_idxs found: {len(convos_idxs)}")

    if not convos_idxs:
        print(f"[WARN] No convos_idxs found for case {case_id}")
        continue

    query_vector = np.mean(embeddings_np[convos_idxs], axis=0).astype('float32').reshape(1, -1)
    print(f"[DEBUG] Query vector shape: {query_vector.shape}")

    support_docs = retrieve_justia_chunks(query_vector, case_id)
    if not support_docs:
        print(f"[WARN] No support docs found for case {case_id}")
        continue

    print(f"[INFO] Retrieved {len(support_docs)} support docs for case {case_id}")

    references_text = "\n".join(
        [f"[{j+1}] {doc.page_content}" for j, doc in enumerate(support_docs)]
    )

    full_source_text = transcripts_text + "\n\n" + justia_text

    prompt_template = (
    f"summarize: Given the following oral argument transcript for case {case_id}, "
    f"create a factual summary grounded in the references.\n\n"
    f"Transcript:\n{{}}\n\n"
    f"Cite the relevant legal opinion excerpts provided below to support the summary.\n"
    f"References (legal opinion excerpts):\n{references_text}\n\n"
    f"Write the summary with numbered citations where relevant."
)
    MAX_INPUT_TOKENS = 512  
    SAFETY_MARGIN = 50

    max_transcript_tokens = MAX_INPUT_TOKENS - SAFETY_MARGIN
    transcript_tokens = tokenizer.tokenize(transcripts_text)

    print(f"[DEBUG] Raw transcript tokens: {len(transcript_tokens)}")

    if len(transcript_tokens) > max_transcript_tokens:
        print(f"[INFO] Trimming transcript from {len(transcript_tokens)} to {max_transcript_tokens} tokens")
        trimmed_tokens = transcript_tokens[:max_transcript_tokens]
        trimmed_transcript = tokenizer.convert_tokens_to_string(trimmed_tokens)
    else:
        trimmed_transcript = transcripts_text

    prompt = prompt_template.format(trimmed_transcript)

    prompt_tokens = len(tokenizer.tokenize(prompt))
    print(f"[DEBUG] Prompt length in tokens AFTER TRIM: {prompt_tokens}")


    # ✅ Use LexT5 for generation
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512  
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n📌 [SUMMARY] Case {case_id}:\n{summary}\n")

    # ✅ Compute ROUGE if gold label exists
    if case_id in gold_labels:
        ref = gold_labels[case_id]
        rouge_score = scorer.score(ref, summary)
        print(f"[ROUGE] for Case {case_id}: {rouge_score}")
    else:
        rouge_score = None

    results.append({
        "case_id": case_id,
        "summary": summary,
        "rouge_score": rouge_score
    })

# ✅ Save results
pd.DataFrame(results).to_csv("data/summaries_and_scores_TEST.csv", index=False)
print("\n✅ Small test results saved to data/summaries_and_scores_TEST.csv")
