import numpy as np
import tiktoken
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 检查 GPU 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_token_logprobs(text: str, model, tokenizer, max_tokens=1024):
    """
    Computes token log probabilities for the given text using the specified model and tokenizer
    """
    try:
        inputs = tokenizer.encode(text, return_tensors="pt", max_length=max_tokens, truncation=True).to(device)
        token_ids = inputs.squeeze(0).cpu().numpy().tolist()
        # 检查 token id 是否超出词表范围
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
        if not all(0 <= tid < vocab_size for tid in token_ids):
            print(f"[Warning] Token id out of range: {token_ids} (vocab_size={vocab_size})")
            return [], np.array([])
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs, labels=inputs)
            logits = outputs.logits
        logprobs = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        probs = np.array([logprobs[i, tid] for i, tid in enumerate(token_ids)], dtype=np.float64)
        return tokens, probs
    except Exception as e:
        print(f"[Error] compute_token_logprobs failed: {e}")
        return [], np.array([])

def write_logprobs(text, file, model, tokenizer):
    """
    Run text under model and write logprobs to file, separated by newline.
    """
    doc = text.strip()
    if not doc or all(c in " \n\t\r" for c in doc):
        print(f"[Warning] Empty or whitespace-only document for file {file}, skipping.")
        return

    if "neo" in file:
        max_tokens = 2048
    elif "gpt2" in file:
        max_tokens = 1024
    else:
        max_tokens = 1024

    # 先 encode 截断，再 decode，确保 decode 后再 encode 长度不会超
    tokens = tokenizer.encode(doc, max_length=max_tokens, truncation=True)
    tokens = tokens[:max_tokens]
    doc = tokenizer.decode(tokens)
    # 再次 encode 检查长度
    tokens_check = tokenizer.encode(doc)
    if len(tokens_check) > max_tokens:
        tokens_check = tokens_check[:max_tokens]
        doc = tokenizer.decode(tokens_check)

    try:
        subwords, subprobs = compute_token_logprobs(doc, model, tokenizer, max_tokens=max_tokens)
        # 类型和长度检查
        if (
            len(subwords) == 0 or len(subprobs) == 0
            or not isinstance(subwords, list)
            or not isinstance(subprobs, np.ndarray)
            or not all(isinstance(w, str) for w in subwords)
            or len(subwords) != len(subprobs)
        ):
            print(f"[Warning] Skipping file {file} due to tokenization/model error or type mismatch.")
            return
        if all(p == 0 for p in subprobs):
            print(f"[Warning] Skipping file {file} because all logprobs are zero.")
            return
    except Exception as e:
        print(f"[Error] {file}: {e}")
        print(f"[Error] Content head: {repr(doc[:100])}")
        return
    gpt2_map = {"\n": "Ċ", "\t": "ĉ", " ": "Ġ"}
    for i in range(len(subwords)):
        w = str(subwords[i])
        for k, v in gpt2_map.items():
            w = w.replace(k, v)
        w = w.replace("\t", "<TAB>").replace("\n", "<NL>")
        subwords[i] = w
    to_write = ""
    for w, p in zip(subwords, subprobs):
        to_write += f"{w}\t{float(p)}\n"
    with open(file, "w", encoding="utf-8") as f:
        f.write(to_write)