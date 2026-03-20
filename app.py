"""
AI Code Assistant — Desktop App
Runs in Spyder (Anaconda Navigator) or any Python terminal

HOW TO RUN IN SPYDER:
    1. Open Anaconda Navigator → Launch Spyder
    2. File → Open → select this file (app.py)
    3. Press F5  (or click the green Run button)
    4. Interact in the bottom-right IPython Console

INSTALL REQUIREMENTS (open Anaconda Prompt / terminal):
    pip install transformers torch faiss-cpu accelerate bitsandbytes einops numpy huggingface_hub

FOLDER STRUCTURE:
    my_project/
    ├── app.py                   ← this file
    └── artifacts/
        ├── faiss_index.bin      ← downloaded from Colab Step 12
        ├── metadata_store.pkl   ← downloaded from Colab Step 12
        └── knowledge_base.pkl   ← downloaded from Colab Step 12
"""

# ── Imports ────────────────────────────────────────────────
import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

import torch
import faiss
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    BitsAndBytesConfig
)

# ── Config ─────────────────────────────────────────────────
ARTIFACTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
CODEBERT_MODEL = 'microsoft/codebert-base'
DEEPSEEK_MODEL = 'deepseek-ai/deepseek-coder-1.3b-instruct'

# ── Globals (loaded once, reused every query) ───────────────
cb_tok      = None
cb_mod      = None
ds_tok      = None
ds_mod      = None
faiss_index = None
metadata    = None
chat_memory = []   # stores conversation history


# ═══════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════

def load_models():
    """Load CodeBERT, DeepSeek Coder, and FAISS index."""
    global cb_tok, cb_mod, ds_tok, ds_mod, faiss_index, metadata

    print('\n' + '=' * 55)
    print('  AI Code Assistant — Loading Models')
    print('=' * 55)

    # ── CodeBERT (Embedding model) ─────────────────────────
    print('\n[1/3] Loading CodeBERT...')
    cb_tok = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
    cb_mod = AutoModel.from_pretrained(CODEBERT_MODEL)
    cb_mod.eval()
    if torch.cuda.is_available():
        cb_mod = cb_mod.cuda()
    print('      OK - CodeBERT ready')

    # ── DeepSeek Coder (LLM) ──────────────────────────────
    print('\n[2/3] Loading DeepSeek Coder 1.3B...')
    print('      (First run downloads ~2.5 GB — takes 3-5 min)')

    bnb_config = None
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    ds_tok = AutoTokenizer.from_pretrained(
        DEEPSEEK_MODEL,
        trust_remote_code=True
    )
    ds_mod = AutoModelForCausalLM.from_pretrained(
        DEEPSEEK_MODEL,
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    ds_mod.eval()
    print('      OK - DeepSeek Coder ready')

    # ── FAISS Index ────────────────────────────────────────
    print('\n[3/3] Loading FAISS index from artifacts/...')

    faiss_path = os.path.join(ARTIFACTS_DIR, 'faiss_index.bin')
    meta_path  = os.path.join(ARTIFACTS_DIR, 'metadata_store.pkl')

    if not os.path.exists(faiss_path):
        print('\nERROR: artifacts/faiss_index.bin not found!')
        print('Please download it from Colab Step 12 and put it in artifacts/')
        sys.exit(1)

    if not os.path.exists(meta_path):
        print('\nERROR: artifacts/metadata_store.pkl not found!')
        print('Please download it from Colab Step 12 and put it in artifacts/')
        sys.exit(1)

    faiss_index = faiss.read_index(faiss_path)
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)

    print(f'      OK - {faiss_index.ntotal} code snippets indexed')

    # ── Print system info ──────────────────────────────────
    print('\n' + '-' * 55)
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'  GPU  : {torch.cuda.get_device_name(0)}')
        print(f'  VRAM : {used:.1f} / {total:.1f} GB')
    else:
        print('  GPU  : Not available (running on CPU)')
    print('-' * 55)
    print('\nAll models loaded! Type any number to start.\n')


# ═══════════════════════════════════════════════════════════
# EMBEDDING + RETRIEVAL (CodeBERT + FAISS)
# ═══════════════════════════════════════════════════════════

def get_embedding(text):
    """Convert any text/code into a 768-dim vector using CodeBERT."""
    tokens = cb_tok(
        text,
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding='max_length'
    )
    if torch.cuda.is_available():
        tokens = {k: v.cuda() for k, v in tokens.items()}
    with torch.no_grad():
        out = cb_mod(**tokens)
    vec = out.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
    return vec / (np.linalg.norm(vec) + 1e-10)


def retrieve(query, top_k=3):
    """Find top-k most relevant snippets for a query using FAISS."""
    q_vec = get_embedding(query).astype('float32').reshape(1, -1)
    scores, indices = faiss_index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            item = metadata[idx].copy()
            item['score'] = float(score)
            results.append(item)
    return results


# ═══════════════════════════════════════════════════════════
# PROMPT ENGINEERING (Mode Templates)
# ═══════════════════════════════════════════════════════════

MODE_PROMPTS = {
    'explain': (
        'You are a coding teacher. Explain the code clearly:\n'
        '1. What it does step by step\n'
        '2. Key concepts and patterns used\n'
        '3. Time/space complexity if algorithmic\n'
        '4. Any edge cases to know about'
    ),
    'fix': (
        'You are a debugging expert. For the given code:\n'
        '1. Find every bug and explain why it is wrong\n'
        '2. Show the complete corrected code\n'
        '3. Briefly explain how to test the fix'
    ),
    'improve': (
        'You are a senior code reviewer. For the given code:\n'
        '1. Point out readability and performance issues\n'
        '2. Show the improved version with inline comments\n'
        '3. Rate the original code out of 10'
    ),
    'answer': (
        'You are a helpful Python programming assistant.\n'
        '1. Answer the question directly and clearly\n'
        '2. Show a short, working code example\n'
        '3. Keep the explanation concise'
    )
}


# ═══════════════════════════════════════════════════════════
# FULL RAG PIPELINE
# ═══════════════════════════════════════════════════════════

def ask(query, mode='answer', top_k=3, max_tokens=512):
    """
    Full RAG pipeline:
      1. RETRIEVE  — CodeBERT + FAISS finds relevant snippets
      2. AUGMENT   — Add snippets to system message
      3. GENERATE  — DeepSeek Coder generates clean answer
      4. MEMORY    — Save turn to conversation history
    """
    t0 = time.time()

    print('\n' + '=' * 55)
    print(f'  Mode  : {mode.upper()}')
    short_q = query[:50] + '...' if len(query) > 50 else query
    print(f'  Query : {short_q}')
    print('=' * 55)

    # ── Step 1: RETRIEVE ──────────────────────────────────
    print('\n[1/3] Retrieving relevant snippets...')
    snippets = retrieve(query, top_k=top_k)
    for s in snippets:
        print(f'      [{s["score"]:.3f}] {s["title"]}')

    # ── Step 2: AUGMENT ───────────────────────────────────
    print('\n[2/3] Building augmented prompt...')

    system_text = MODE_PROMPTS.get(mode, MODE_PROMPTS['answer'])

    # Inject retrieved snippets into system message (the A in RAG)
    if snippets:
        system_text += '\n\nRelevant code examples for reference:\n'
        for i, s in enumerate(snippets, 1):
            pct = int(s['score'] * 100)
            system_text += f'\n[{i}] {s["title"]} ({pct}% match)\n'
            system_text += f'```python\n{s["code"]}\n```\n'
            system_text += f'Note: {s["note"]}\n'

    # Build message list for DeepSeek
    messages = [{'role': 'system', 'content': system_text}]

    # Add recent conversation history (memory)
    if chat_memory:
        messages.extend(chat_memory[-6:])  # last 3 turns

    # Add current user query
    messages.append({'role': 'user', 'content': query})

    # ✅ Correct DeepSeek tokenization
    # Step 1: build the prompt string using apply_chat_template
    # Step 2: tokenize it separately with ds_tok() to get a clean tensor
    prompt_str = ds_tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False         # get plain string first
    )
    # Step 3: tokenize the string → guaranteed plain tensor, no Encoding issues
    tokenized = ds_tok(
        prompt_str,
        return_tensors='pt',
        truncation=True,
        max_length=2048
    )
    input_ids = tokenized['input_ids']

    print(f'      Prompt = {input_ids.shape[1]} tokens')

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    # ── Step 3: GENERATE ──────────────────────────────────
    print('\n[3/3] Generating with DeepSeek Coder...')

    eot_id   = ds_tok.convert_tokens_to_ids('<|EOT|>')
    stop_ids = [ds_tok.eos_token_id]
    if eot_id is not None:
        stop_ids.append(eot_id)

    with torch.no_grad():
        out = ds_mod.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,           # greedy = clean stable output
            eos_token_id=stop_ids,
            pad_token_id=ds_tok.eos_token_id
        )

    new_ids = out[0][input_ids.shape[1]:]
    # Decode and clean BPE artifacts (G with dot = space, C with cedilla = newline)
    answer = ds_tok.batch_decode([new_ids], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
    answer = answer.replace("\u0120", " ").replace("\u010a", "\n").strip()

    # ── Step 4: MEMORY ────────────────────────────────────
    chat_memory.append({'role': 'user',      'content': query})
    chat_memory.append({'role': 'assistant', 'content': answer})
    # Keep only last 5 turns (10 messages)
    if len(chat_memory) > 10:
        chat_memory.pop(0)
        chat_memory.pop(0)

    elapsed = time.time() - t0
    print(f'      Done in {elapsed:.1f}s | {len(new_ids)} tokens generated')

    print('\n' + '-' * 55)
    print('ANSWER:')
    print('-' * 55)
    print(answer)
    print('-' * 55)

    return answer


# ═══════════════════════════════════════════════════════════
# INTERACTIVE MENU (Spyder IPython Console)
# ═══════════════════════════════════════════════════════════

def get_multiline_input(prompt_text):
    """
    Collect multi-line input in Spyder's IPython console.
    User types their code/question line by line.
    Type END on a new line to finish.
    """
    print(prompt_text)
    print('  Paste or type your input below.')
    print('  When finished, type  END  on a new line and press Enter.')
    print('-' * 55)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == 'END':
            break
        lines.append(line)
    result = '\n'.join(lines).strip()
    return result


def show_menu():
    print('\n' + '=' * 55)
    print('  AI Code Assistant — Menu')
    print('=' * 55)
    print('  1  Ask a coding question    (Q&A)')
    print('  2  Explain code             (Explain)')
    print('  3  Fix buggy code           (Fix)')
    print('  4  Improve code             (Improve)')
    print('  5  Show conversation history')
    print('  6  Clear conversation history')
    print('  0  Exit')
    print('-' * 55)


def show_history():
    if not chat_memory:
        print('\n  No history yet.')
        return
    print(f'\n  Conversation History ({len(chat_memory) // 2} turns):')
    print('-' * 55)
    for t in chat_memory:
        who  = 'You' if t['role'] == 'user' else 'Bot'
        text = t['content']
        if len(text) > 120:
            text = text[:120] + '...'
        print(f'\n  [{who}]\n  {text}')
    print('-' * 55)


def run():
    """
    Main app loop.
    Call run() from Spyder or just press F5 — it auto-starts.
    """
    load_models()

    while True:
        show_menu()

        try:
            choice = input('  Your choice: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n\nGoodbye!')
            break

        if choice == '0':
            print('\nGoodbye!')
            break

        elif choice == '1':
            query = get_multiline_input('\n  Q&A — Enter your question:')
            if query:
                ask(query, mode='answer')
            else:
                print('  No input entered.')

        elif choice == '2':
            query = get_multiline_input('\n  EXPLAIN — Paste the code you want explained:')
            if query:
                ask(query, mode='explain')
            else:
                print('  No input entered.')

        elif choice == '3':
            query = get_multiline_input('\n  FIX — Paste the buggy code:')
            if query:
                ask(query, mode='fix')
            else:
                print('  No input entered.')

        elif choice == '4':
            query = get_multiline_input('\n  IMPROVE — Paste the code to improve:')
            if query:
                ask(query, mode='improve')
            else:
                print('  No input entered.')

        elif choice == '5':
            show_history()

        elif choice == '6':
            chat_memory.clear()
            print('\n  Conversation history cleared.')

        else:
            print('\n  Invalid choice. Enter 0 to 6.')


# ── Auto-start when F5 is pressed in Spyder ────────────────
if __name__ == '__main__':
    run()
