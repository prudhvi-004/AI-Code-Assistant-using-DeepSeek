import streamlit as st
import torch
import faiss
import pickle
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

# ── API SETUP ───────────────────────────────────────────
api_key = st.secrets.get("OPENROUTER_API_KEY")

if not api_key:
    st.error("❌ API key missing. Add OPENROUTER_API_KEY in Streamlit secrets.")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# ── SESSION ─────────────────────────────────────────────
for k, v in dict(
    loaded=False,
    history=[],
    cb_tok=None,
    cb_mod=None,
    faiss_idx=None,
    metadata=None
).items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── LOADERS ─────────────────────────────────────────────
@st.cache_resource
def load_codebert():
    tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    mod = AutoModel.from_pretrained("microsoft/codebert-base")
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

def generate(query, snippets):
    context = ""
    for s in snippets:
        context += f"{s['code']}\n"

    prompt = f"""
You are an expert coding assistant.

Context:
{context}

User Query:
{query}

Instructions:
- Give clear answer
- Provide code if needed
- Keep concise
"""

    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",   # ✅ FIXED MODEL
        messages=[
            {"role": "system", "content": "You are a coding assistant"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# ── UI ─────────────────────────────────────────────────
st.set_page_config(page_title="AI Code Assistant", page_icon="🤖")

st.title("🤖 AI Code Assistant (DeepSeek + RAG)")
st.markdown("CodeBERT + FAISS + DeepSeek API")

# Load models
if not st.session_state.loaded:
    if st.button("🚀 Load Models"):
        with st.spinner("Loading CodeBERT..."):
            st.session_state.cb_tok, st.session_state.cb_mod = load_codebert()
        with st.spinner("Loading FAISS..."):
            st.session_state.faiss_idx, st.session_state.metadata = load_faiss()
        st.session_state.loaded = True
        st.success("✅ Ready!")
        st.rerun()

query = st.text_area("Enter your code or question", height=200)

if st.button("⚡ Generate Answer"):
    if not st.session_state.loaded:
        st.error("Load models first")
    elif not query.strip():
        st.warning("Enter a question")
    else:
        with st.spinner("🔍 Retrieving..."):
            snippets = retrieve(
                query,
                st.session_state.faiss_idx,
                st.session_state.metadata,
                st.session_state.cb_tok,
                st.session_state.cb_mod
            )

        with st.spinner("🧠 DeepSeek thinking..."):
            answer = generate(query, snippets)

        st.markdown("## ✅ Answer")
        st.write(answer)

        st.session_state.history.append({"q": query, "a": answer})

# History
if st.session_state.history:
    with st.expander("🕘 History"):
        for item in reversed(st.session_state.history):
            st.markdown(f"**You:** {item['q']}")
            st.markdown(f"**Bot:** {item['a']}")
            st.markdown("---")
