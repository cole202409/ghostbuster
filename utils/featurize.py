import numpy as np
import os
import tqdm
import re
from nltk import ngrams
from utils.score import k_fold_score
import glob

def safe_trunc(arr, n):
    return arr[:n] if len(arr) > n else arr

def get_logprobs(file):
    """
    Returns a vector containing all the logprobs from a given logprobs file
    """
    logprobs = []
    bad_line_count = 0
    with open(file) as f:
        for idx, line in enumerate(f.read().strip().split("\n")):
            # 用正则分割，支持多个空格或制表符
            parts = re.split(r"[\t ]+", line.strip())
            if len(parts) < 2:
                # 跳过空行或格式错误行
                if len(parts) == 1 and parts[0] == "":
                    continue
                bad_line_count += 1
                continue
            try:
                logprobs.append(float(parts[1]))
            except Exception as e:
                bad_line_count += 1
                continue
    if bad_line_count > 0:
        print(f"[Warning] Skipped file {file} due to {bad_line_count} bad lines.")
        return np.array([])  # 直接返回空数组，后续流程会自动跳过
    return np.array(logprobs)

def get_tokens(file):
    """
    Returns a list of all tokens from a given logprobs file
    """
    tokens = []
    bad_line_count = 0
    with open(file) as f:
        for idx, line in enumerate(f.read().strip().split("\n")):
            parts = re.split(r"[\t ]+", line.strip())
            if len(parts) < 2:
                # 跳过空行或格式错误行
                if len(parts) == 1 and parts[0] == "":
                    continue
                bad_line_count += 1
                continue
            tokens.append(parts[0])
    if bad_line_count > 0:
        print(f"[Warning] Skipped file {file} due to {bad_line_count} bad lines.")
        return []  # 直接返回空列表，后续流程会自动跳过
    return tokens

def get_token_len(tokens):
    """
    Returns a list of token lengths, considering GPT-2 style tokens
    """
    token_len = []
    curr = 1
    for token in tokens:
        if len(token) > 0 and token[0] == "Ġ":
            token_len.append(curr)
            curr = 1
        else:
            curr += 1
    if curr > 1:
        token_len.append(curr)
    return np.array(token_len)

def get_diff(file1, file2):
    """
    Returns difference in logprobs between file1 and file2
    """
    return get_logprobs(file1) - get_logprobs(file2)

def convolve(X, window=100):
    """
    Returns a vector of running average with window size
    """
    ret = []
    for i in range(len(X) - window):
        ret.append(np.mean(X[i : i + window]))
    return np.array(ret)

def score_ngram(doc, model, tokenize, n=3, strip_first=False):
    """
    Returns vector of ngram log probabilities given document, model and tokenizer
    """
    tokens = tokenize(doc.strip())
    # 保证 tokens 是 list of int
    if isinstance(tokens, int):
        tokens = [tokens]
    elif isinstance(tokens, tuple):
        tokens = list(tokens)
    elif not isinstance(tokens, list):
        if hasattr(tokens, "input_ids"):
            tokens = tokens["input_ids"]
            if isinstance(tokens, list):
                tokens = tokens[0]
            else:
                tokens = tokens.squeeze().tolist()
        else:
            tokens = list(tokens)
    # 递归展开嵌套
    while isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
        tokens = tokens[0]
    # 最终 tokens 必须是 list of int
    tokens = [int(t) for t in tokens]
    scores = []
    for i in ngrams((n - 1) * [50256] + tokens, n):
        prob = model.get(i, 1e-10)  # 平滑处理
        scores.append(np.log(prob))
    if strip_first and scores:
        scores = scores[1:]
    return np.array(scores)

def normalize(data, mu=None, sigma=None, ret_mu_sigma=False):
    """
    Normalizes data, where data is a matrix where the first dimension is the number of examples
    """
    if mu is None:
        mu = np.mean(data, axis=0)
    if sigma is None:
        raw_std = np.std(data, axis=0)
        sigma = np.ones_like(raw_std)
        sigma[raw_std != 0] = raw_std[raw_std != 0]

    if ret_mu_sigma:
        return (data - mu) / sigma, mu, sigma
    else:
        return (data - mu) / sigma

def convert_file_to_logprob_file(file_name, model):
    """
    支持递归向上查找所有父目录下的 logprobs 文件夹，优先返回实际存在的文件。
    """
    directory = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    file_name_without_ext = os.path.splitext(base_name)[0]
    logprob_file_name = f"{file_name_without_ext}-{model}.txt"
    # 递归向上查找 logprobs 目录
    cur_dir = directory
    while True:
        candidate = os.path.join(cur_dir, "logprobs", logprob_file_name)
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(cur_dir)
        if parent == cur_dir:
            break
        cur_dir = parent
    # 兼容 reuter 结构：如 ghostbuster-data/reuter/gpt/某子文件夹/logprobs
    # 查找所有子目录下的 logprobs
    for root, dirs, files in os.walk(directory):
        if "logprobs" in dirs:
            candidate = os.path.join(root, "logprobs", logprob_file_name)
            if os.path.exists(candidate):
                return candidate
    # 默认返回原有规则
    logprob_directory = os.path.join(directory, "logprobs")
    logprob_file_path = os.path.join(logprob_directory, logprob_file_name)
    return logprob_file_path

def t_featurize_logprobs(neo_lp, gpt2_lp, tokens):
    """
    Handcrafted features for classification based on logprobs and tokens
    """
    feats = []

    # 处理空输入
    if len(neo_lp) == 0 or len(gpt2_lp) == 0:
        return np.zeros(7)

    # 异常值处理
    neo_lp = np.clip(neo_lp, -10, 0)
    gpt2_lp = np.clip(gpt2_lp, -10, 0)

    # 提取异常值
    neo_outliers = neo_lp[neo_lp < -5]
    gpt2_outliers = gpt2_lp[gpt2_lp < -5]
    neo_outliers = safe_trunc(neo_outliers, 50)
    gpt2_outliers = safe_trunc(gpt2_outliers, 50)

    # 添加特征
    feats.extend([
        len(neo_outliers),
        np.mean(neo_outliers[:25]) if len(neo_outliers) > 0 else 0.0,
        np.mean(neo_outliers[25:50]) if len(neo_outliers) > 0 else 0.0,
        len(gpt2_outliers),
        np.mean(gpt2_outliers[:25]) if len(gpt2_outliers) > 0 else 0.0,
        np.mean(gpt2_outliers[25:50]) if len(gpt2_outliers) > 0 else 0.0
    ])

    # Token 长度特征
    token_len = [len(t) for t in tokens if isinstance(t, str)]
    token_len = safe_trunc(sorted(token_len, reverse=True), 50)
    feats.append(np.mean(token_len[:25]) if len(token_len) > 0 else 0.0)

    return np.array(feats)

def t_featurize(file, num_tokens=2048):
    try:
        neo_file = convert_file_to_logprob_file(file, "neo")
        gpt2_file = convert_file_to_logprob_file(file, "gpt2")

        if not os.path.exists(neo_file) or not os.path.exists(gpt2_file):
            return np.zeros(7)

        neo_logprobs = get_logprobs(neo_file)[:num_tokens]
        gpt2_logprobs = get_logprobs(gpt2_file)[:num_tokens]
        tokens = get_tokens(neo_file)[:num_tokens]

        result = t_featurize_logprobs(neo_logprobs, gpt2_logprobs, tokens)
        return result
    except Exception as e:
        return np.zeros(7)

def select_features(exp_to_data, labels, verbose=True, to_normalize=True, indices=None, min_improve=0.0, max_features=9):
    if to_normalize:
        normalized_exp_to_data = {}
        for key in exp_to_data:
            normalized_exp_to_data[key] = normalize(exp_to_data[key])
    else:
        normalized_exp_to_data = exp_to_data

    def get_data(*exp):
        return np.concatenate([normalized_exp_to_data[e] for e in exp], axis=1)

    val_exp = list(exp_to_data.keys())
    curr = 0
    best_features = []
    i = 0

    while val_exp and len(best_features) < max_features:
        best_score, best_exp = -1, ""

        for exp in tqdm.tqdm(val_exp) if verbose else val_exp:
            score = k_fold_score(
                get_data(*best_features, exp), labels, k=5, indices=indices
            )

            if score > best_score:
                best_score = score
                best_exp = exp

        if verbose:
            print(
                f"Iteration {i}, Current Score: {curr}, \
                Best Feature: {best_exp}, New Score: {best_score}"
            )
        # 只要没到max_features就继续
        if best_score - curr <= min_improve and len(best_features) > 0:
            break
        else:
            best_features.append(best_exp)
            val_exp.remove(best_exp)
            curr = best_score

        i += 1

    return best_features
