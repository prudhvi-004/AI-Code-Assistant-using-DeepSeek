"""
AI Code Assistant — Streamlit App (Stable Version)
Stack: CodeBERT + FAISS + FLAN-T5 (Lightweight)
"""

import streamlit as st
import torch
import faiss
import pickle
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel

# ── CONFIG ─────────────────────────────────────────────
MODEL_NAME = "google/flan-t5-base"  # ✅ lightweight & fast

st.set_page_config(page_title="AI Code Assistant", page_icon="🤖", layout="wide")

# ── SESSION STATE ──────────────────────────────────────
for k, v in dict(
    loaded=False,
    history=[],
    cb_tok=None,
    cb_mod=None,
    llm_tok=None,
    llm_mod=None,
    faiss_idx=None,
    metadata=None
).items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── LOADERS ────────────────────────────────────────────
@st.cache_resource
def load_codebert():
    tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    mod = AutoModel.from_pretrained("microsoft/codebert-base")
    mod.eval()
    return tok, mod

@st.cache_resource
def load_llm():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mod = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    mod.eval()
    return tok, mod

@st.cache_resource
def load_faiss():
    p1 = Path("artifacts/faiss_index.bin")
    p2 = Path("artifacts/metadata_store.pkl")

    if p1.exists() and p2.exists():
        idx = faiss.read_index(str(p1))
        meta = pickle.load(open(p2, "rb"))
        return idx, meta
    return None, None

# ── CORE FUNCTIONS ─────────────────────────────────────
def get_embedding(text, tok, mod):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = mod(**inputs)
    vec = outputs.last_hidden_state[:, 0, :].numpy().squeeze()
    return vec / (np.linalg.norm(vec) + 1e-10)

def retrieve(query, idx, meta, tok, mod, top_k=3):
    if idx is None:
        return []
    q_vec = get_embedding(query, tok, mod).astype("float32").reshape(1, -1)
    scores, indices = idx.search(q_vec, top_k)

    results = []
    for score, i in zip(scores[0], indices[0]):
        if i >= 0:
            item = meta[i].copy()
            item["score"] = float(score)
            results.append(item)
    return results

def generate(query, snippets, tok, mod):
    context = ""
    for s in snippets:
        context += f"{s['code']}\n"

    prompt = f"""
You are a helpful coding assistant.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tok(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = mod.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=150
        )

    answer = tok.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

# ── UI ────────────────────────────────────────────────
st.title("🤖 AI Code Assistant")
st.markdown("### CodeBERT + FAISS + FLAN-T5 (Fast & Stable)")

# Load models
if not st.session_state.loaded:
    if st.button("🚀 Load Models"):
        with st.spinner("Loading models..."):
            st.session_state.cb_tok, st.session_state.cb_mod = load_codebert()
            st.session_state.llm_tok, st.session_state.llm_mod = load_llm()
            st.session_state.faiss_idx, st.session_state.metadata = load_faiss()
            st.session_state.loaded = True
        st.success("Models loaded!")
        st.rerun()

# Input
query = st.text_area("Enter your code or question", height=200)

# Generate
if st.button("⚡ Generate Answer"):
    if not st.session_state.loaded:
        st.error("Please load models first.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🔍 Retrieving context..."):
            snippets = retrieve(
                query,
                st.session_state.faiss_idx,
                st.session_state.metadata,
                st.session_state.cb_tok,
                st.session_state.cb_mod
            )

        with st.spinner("🧠 Generating answer..."):
            answer = generate(
                query,
                snippets,
                st.session_state.llm_tok,
                st.session_state.llm_mod
            )

        st.markdown("## ✅ Answer")
        st.write(answer)

        # Save history
        st.session_state.history.append({"q": query, "a": answer})

# History
if st.session_state.history:
    with st.expander("🕘 History"):
        for item in reversed(st.session_state.history):
            st.markdown(f"**You:** {item['q']}")
            st.markdown(f"**Bot:** {item['a']}")
            st.markdown("---")
