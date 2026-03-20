"""
AI Code Assistant — Streamlit App
Stack: DeepSeek Coder + CodeBERT + FAISS + RAG
Deploy on: https://streamlit.io/cloud
"""

import streamlit as st
import torch
import faiss
import pickle
import numpy as np
import time
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    BitsAndBytesConfig,
)

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Code Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap');

html, body, .stApp        { background:#0b0b14 !important; color:#dddaf5 !important; }
h1,h2,h3                  { font-family:'Space Grotesk',sans-serif !important; }
.stTextArea textarea       { background:#14142a !important; border:1px solid #2a2a50 !important;
                             color:#dddaf5 !important; border-radius:8px !important;
                             font-family:'JetBrains Mono',monospace !important; }
.stButton>button           { background:#00b4d8 !important; color:#000 !important;
                             border:none !important; border-radius:8px !important;
                             font-family:'Space Grotesk',sans-serif !important; font-weight:700 !important; }
.stButton>button:hover     { background:#0096c7 !important; }
section[data-testid="stSidebar"] { background:#0d0d1e !important; border-right:1px solid #2a2a50 !important; }
.tag  { display:inline-block; background:#141430; border:1px solid #333366;
        border-radius:20px; padding:3px 12px; font-size:0.72rem;
        font-family:'JetBrains Mono',monospace; color:#8888cc; margin:2px; }
.tag-ok  { border-color:#00b894; color:#00b894; }
.tag-ds  { border-color:#00b4d8; color:#00b4d8; }
footer, #MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────
for k, v in dict(
    loaded=False, history=[],
    cb_tok=None, cb_mod=None,
    ds_tok=None, ds_mod=None,
    faiss_idx=None, metadata=None,
    mode="answer"
).items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Model Loaders ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_codebert():
    tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    mod = AutoModel.from_pretrained("microsoft/codebert-base")
    mod.eval()
    if torch.cuda.is_available():
        mod = mod.cuda()
    return tok, mod

@st.cache_resource(show_spinner=False)
def load_deepseek():
    bnb = None
    if torch.cuda.is_available():
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    tok = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        trust_remote_code=True
    )
    mod = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        quantization_config=bnb,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    mod.eval()
    return tok, mod

@st.cache_resource(show_spinner=False)
def load_faiss():
    p1 = Path("artifacts/faiss_index.bin")
    p2 = Path("artifacts/metadata_store.pkl")
    if p1.exists() and p2.exists():
        idx  = faiss.read_index(str(p1))
        meta = pickle.load(open(p2, "rb"))
        return idx, meta
    return None, None

# ── Core Functions ───────────────────────────────────────────
def get_embedding(text, tok, mod):
    inp = tok(text, return_tensors="pt", max_length=512,
              truncation=True, padding="max_length")
    if torch.cuda.is_available():
        inp = {k: v.cuda() for k, v in inp.items()}
    with torch.no_grad():
        out = mod(**inp)
    vec = out.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
    return vec / (np.linalg.norm(vec) + 1e-10)

def retrieve(query, idx, meta, tok, mod, top_k=3):
    if idx is None:
        return []
    q = get_embedding(query, tok, mod).astype("float32").reshape(1, -1)
    scores, ids = idx.search(q, top_k)
    return [
        {**meta[i], "score": float(s)}
        for s, i in zip(scores[0], ids[0]) if i >= 0
    ]

MODE_PROMPTS = {
    "explain": (
        "You are a coding teacher. Explain the code clearly:\n"
        "1. What it does step by step\n"
        "2. Key concepts and patterns used\n"
        "3. Time/space complexity if algorithmic\n"
        "4. Any edge cases to know about"
    ),
    "fix": (
        "You are a debugging expert. For the given code:\n"
        "1. Find every bug and explain why it is wrong\n"
        "2. Show the complete corrected code\n"
        "3. Briefly explain how to test the fix"
    ),
    "improve": (
        "You are a senior code reviewer. For the given code:\n"
        "1. Point out readability and performance issues\n"
        "2. Show the improved version with inline comments\n"
        "3. Rate the original code out of 10"
    ),
    "answer": (
        "You are a helpful Python programming assistant.\n"
        "1. Answer the question directly and clearly\n"
        "2. Show a short, working code example\n"
        "3. Keep the explanation concise"
    )
}

def build_prompt(query, snippets, mode, history):
    system = MODE_PROMPTS.get(mode, MODE_PROMPTS["answer"])
    if snippets:
        system += "\n\nRelevant code examples:\n"
        for i, s in enumerate(snippets, 1):
            pct = int(s["score"] * 100)
            system += f"\n[{i}] {s['title']} ({pct}% match)\n"
            system += f"```python\n{s['code']}\n```\n"
            system += f"Note: {s['note']}\n"

    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": query})
    return messages

def generate(query, mode, snippets, history, tok, mod, max_tokens=400):
    messages = build_prompt(query, snippets, mode, history)

    # Build prompt string using chat template
    prompt_str = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # Tokenize cleanly
    tokenized = tok(
        prompt_str,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )
    input_ids = tokenized["input_ids"]

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    eot_id   = tok.convert_tokens_to_ids("<|EOT|>")
    stop_ids = [tok.eos_token_id]
    if eot_id is not None:
        stop_ids.append(eot_id)

    with torch.no_grad():
        out = mod.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=stop_ids,
            pad_token_id=tok.eos_token_id
        )

    new_ids = out[0][input_ids.shape[1]:]

    # ✅ Fixed decoding — removes Ġ and Ċ BPE artifacts
    answer = tok.batch_decode(
        [new_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0].strip()
    answer = answer.replace("\u0120", " ").replace("\u010a", "\n").strip()

    return answer

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 AI Code Assistant")
    st.markdown("<span class='tag tag-ds'>DeepSeek Coder</span>", unsafe_allow_html=True)
    st.markdown("---")

    gpu = torch.cuda.is_available()
    st.markdown(
        f"<span class='tag tag-{'ok' if gpu else 'ds'}'>{'🟢 GPU' if gpu else '🟡 CPU'}</span>",
        unsafe_allow_html=True
    )

    if not st.session_state.loaded:
        if st.button("🚀 Load Models", use_container_width=True):
            with st.spinner("Loading CodeBERT..."):
                t, m = load_codebert()
                st.session_state.cb_tok = t
                st.session_state.cb_mod = m
            with st.spinner("Loading DeepSeek Coder..."):
                t, m = load_deepseek()
                st.session_state.ds_tok = t
                st.session_state.ds_mod = m
            with st.spinner("Loading FAISS index..."):
                idx, meta = load_faiss()
                st.session_state.faiss_idx = idx
                st.session_state.metadata  = meta
            st.session_state.loaded = True
            st.rerun()
    else:
        st.markdown("<span class='tag tag-ok'>✅ Models Ready</span>", unsafe_allow_html=True)
        if st.session_state.faiss_idx:
            n = st.session_state.faiss_idx.ntotal
            st.markdown(f"<span class='tag'>📚 {n} snippets</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚙️ Settings**")
    max_tok = st.slider("Max tokens",     100, 800, 400, 50)
    top_k   = st.slider("Context snippets", 1,   5,   3)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Turns:** {len(st.session_state.history)//2}")
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.markdown("---")
    st.markdown("**Stack**")
    for tag in ["🧠 DeepSeek Coder", "🔢 CodeBERT", "🗂️ FAISS", "📚 RAG", "💾 Pickle"]:
        st.markdown(f"<span class='tag'>{tag}</span>", unsafe_allow_html=True)

# ── Main ─────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;"
    "background:linear-gradient(90deg,#00b4d8,#48cae4);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent'>"
    "🤖 AI Code Assistant</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:#6666aa'>"
    "Explain · Fix · Improve · Answer — DeepSeek Coder + RAG + CodeBERT + FAISS</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# Mode buttons
st.markdown("**Choose Mode:**")
c1, c2, c3, c4 = st.columns(4)
MODES = {
    "⚡ Explain": "explain",
    "🐛 Fix Bug": "fix",
    "✨ Improve": "improve",
    "💬 Q&A":    "answer"
}
for col, (label, val) in zip([c1, c2, c3, c4], MODES.items()):
    with col:
        if st.button(label, use_container_width=True,
                     type="primary" if st.session_state.mode == val else "secondary"):
            st.session_state.mode = val
            st.rerun()

st.markdown("")

EXAMPLES = {
    "explain": "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\n\nprint(fib(10))",
    "fix":     "def calc(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total / len(nums)\n\nprint(calc([]))  # crashes!",
    "improve": "def get_big(lst):\n    result = []\n    for i in range(len(lst)):\n        if lst[i] > 10:\n            result.append(lst[i])\n    return result",
    "answer":  "What is the difference between a list and a tuple in Python?"
}

_, col_ex = st.columns([6, 1])
with col_ex:
    if st.button("📋 Example", use_container_width=True):
        st.session_state["_pre"] = EXAMPLES[st.session_state.mode]
        st.rerun()

prefill    = st.session_state.pop("_pre", "")
user_input = st.text_area(
    f"Your code or question — **{st.session_state.mode}** mode:",
    value=prefill,
    height=180,
    placeholder=EXAMPLES[st.session_state.mode]
)

run_btn = st.button("⚡ Generate Answer", use_container_width=True, type="primary")

# ── Run ──────────────────────────────────────────────────────
if run_btn and user_input.strip():
    if not st.session_state.loaded:
        st.error("⚠️ Click Load Models in the sidebar first.")
    else:
        with st.spinner("🔍 Searching with CodeBERT + FAISS..."):
            snippets = retrieve(
                user_input,
                st.session_state.faiss_idx,
                st.session_state.metadata,
                st.session_state.cb_tok,
                st.session_state.cb_mod,
                top_k
            )
        with st.spinner("🧠 Generating with DeepSeek Coder..."):
            answer = generate(
                user_input,
                st.session_state.mode,
                snippets,
                st.session_state.history,
                st.session_state.ds_tok,
                st.session_state.ds_mod,
                max_tok
            )
        st.session_state.history.append({"role": "user",      "content": user_input})
        st.session_state.history.append({"role": "assistant", "content": answer})

# ── Show Result ──────────────────────────────────────────────
if st.session_state.history:
    last = next(
        (t for t in reversed(st.session_state.history) if t["role"] == "assistant"),
        None
    )
    if last:
        st.markdown("---")
        st.markdown("**🤖 Answer:**")
        st.markdown(last["content"])

# ── History ──────────────────────────────────────────────────
if len(st.session_state.history) > 2:
    with st.expander(f"🕘 History ({len(st.session_state.history)//2} turns)"):
        pairs = list(zip(
            st.session_state.history[::2],
            st.session_state.history[1::2]
        ))
        for q, a in reversed(pairs[:-1]):
            st.markdown(f"**You:** {q['content'][:100]}...")
            st.markdown(f"**Bot:** {a['content'][:200]}...")
            st.markdown("---")

# ── Welcome ──────────────────────────────────────────────────
if not st.session_state.loaded:
    st.info(
        "👈 Click **Load Models** in the sidebar to get started.\n\n"
        "**How it works:**\n"
        "1. 🔢 CodeBERT embeds your question into a vector\n"
        "2. 🗂️ FAISS finds the most similar code snippets\n"
        "3. 🧠 DeepSeek Coder reads snippets + question → generates answer\n"
        "4. 💾 Memory keeps the conversation context"
    )
