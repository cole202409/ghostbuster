import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tiktoken
import torch
import os
import argparse
import dill as pickle

from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.featurize import normalize, t_featurize, t_featurize_logprobs, score_ngram
from utils.symbolic import train_trigram, get_words, vec_functions, scalar_functions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_token_logprobs(text: str, model, tokenizer, max_tokens=1024):
    """
    Computes token log probabilities for the given text using the specified model and tokenizer.
    """
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=max_tokens, truncation=True).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        logits = outputs.logits
    logprobs = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    token_ids = inputs.squeeze(0).cpu().numpy().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    probs = np.array([logprobs[i, tid] for i, tid in enumerate(token_ids)], dtype=np.float64)

    return tokens, probs

def load_gpt2_model(local_dir="local_models/gpt2"):
    if not os.path.exists(local_dir):
        print("首次运行：正在下载 GPT2 模型和 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=local_dir)
    model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=local_dir).to(device)
    print("成功加载 GPT2 模型和 tokenizer（来自缓存或网络）")
    return tokenizer, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="input.txt", help="待分类的文本文件路径")
    args = parser.parse_args()

    # —— 1. 加载分类器及特征参数
    clf = pickle.load(open("model/model", "rb"))
    mu = pickle.load(open("model/mu", "rb"))
    sigma = pickle.load(open("model/sigma", "rb"))
    best_features = open("model/features.txt", encoding="utf-8").read().splitlines()

    # —— 2. 文本读取与截断
    raw = open(args.file, encoding="utf-8").read().strip()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tok_ids = tokenizer.encode(raw, max_length=2047, truncation=True)
    doc = tokenizer.decode(tok_ids).strip()

    # —— 3. n-gram 特征
    trigram_model = train_trigram()
    tri_scores = np.array(score_ngram(doc, trigram_model, tokenizer.encode, n=3, strip_first=False))
    uni_scores = np.array(score_ngram(doc, trigram_model.base, tokenizer.encode, n=1, strip_first=False))

    # —— 4. 加载并计算两个模型的 log-probs
    neo_tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    neo_mdl = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
    gpt2_tok, gpt2_mdl = load_gpt2_model()
    tokens, neo_lp = compute_token_logprobs(doc, neo_mdl, neo_tok, max_tokens=2048)
    gpt2_tokens, gpt2_lp = compute_token_logprobs(doc, gpt2_mdl, gpt2_tok, max_tokens=1024)

    # 截断所有序列到同一长度
    L = min(len(tokens), len(gpt2_tokens))
    neo_lp = neo_lp[:L]
    gpt2_lp = gpt2_lp[:L]
    tri_scores = tri_scores[:L]
    uni_scores = uni_scores[:L]
    tokens = tokens[:L]

    # —— 5. 特征化流水线
    gpt2_map = {"\n": "Ċ", "\t": "ĉ", " ": "Ġ"}
    tokens = [t for t in tokens]
    for k, v in gpt2_map.items():
        tokens = [t.replace(k, v) for t in tokens]

    # 1. 生成 t_feats（直接用 logprobs/tokens，保证与训练时一致）
    t_feats = t_featurize_logprobs(neo_lp, gpt2_lp, tokens)
    t_feat_names = [f"t_feat_{i}" for i in range(len(t_feats))]

    # 2. 生成 logprobs/trigram/unigram 分数（和训练时一致）
    neo_logprobs = neo_lp
    gpt2_logprobs = gpt2_lp
    trigram_logprobs = tri_scores
    unigram_logprobs = uni_scores
    vector_map = {
        "neo-logprobs": neo_logprobs,
        "j6b-logprobs": gpt2_logprobs,
        "trigram-logprobs": trigram_logprobs,
        "unigram-logprobs": unigram_logprobs,
    }

    # 3. 读取 all_feat_names.txt，分离 exp_keys
    with open("model/all_feat_names.txt", encoding="utf-8") as f:
        all_feat_names = [line.strip() for line in f if line.strip()]
    exp_keys = all_feat_names[len(t_feats):]  # t_feats 在前，exp_keys 在后

    # 4. 用 get_exp_featurize 生成 exp_feats，保证和训练时完全一致
    from utils.symbolic import get_exp_featurize
    exp_featurize = get_exp_featurize(exp_keys, vector_map)
    exp_feats = exp_featurize(doc)

    # 5. 拼接特征
    all_feats = np.concatenate([t_feats, exp_feats])

    # —— 6. 合并、标准化、预测
    with open("model/all_feat_names.txt", encoding="utf-8") as f:
        all_feat_names = [line.strip() for line in f if line.strip()]
    with open("model/features.txt", encoding="utf-8") as f:
        valid_feat_names = [line.strip() for line in f if line.strip()]
    name2idx = {name: i for i, name in enumerate(all_feat_names)}
    selected_idx = [name2idx[name] for name in valid_feat_names]
    all_feats = all_feats[selected_idx]

    if all_feats.shape[0] != mu.shape[0]:
        raise RuntimeError(f"Feature size mismatch: got {all_feats.shape[0]}, expected {mu.shape[0]}")

    X = (all_feats - mu) / sigma
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    prob = clf.predict_proba(X.reshape(1, -1))[:, 1][0]
    print(f"预测为正例的概率：{prob:.6f}")

if __name__ == "__main__":
    main()
