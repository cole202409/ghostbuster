from nltk.util import ngrams
from nltk.corpus import brown
from nltk.tokenize import word_tokenize

import tqdm
import numpy as np
import os
import tiktoken
import dill as pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from multiprocessing import Pool
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression

from utils.featurize import score_ngram
from utils.n_gram import TrigramBackoff
from utils.write_logprobs import write_logprobs

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义向量函数
vec_functions = {
    "v-add": lambda a, b: np.array(a, dtype=float) + np.array(b, dtype=float),
    "v-sub": lambda a, b: np.array(a, dtype=float) - np.array(b, dtype=float),
    "v-mul": lambda a, b: np.array(a, dtype=float) * np.array(b, dtype=float),
    "v-div": lambda a, b: np.divide(
        np.array(a, dtype=float),
        np.array(b, dtype=float),
        out=np.zeros_like(np.array(a, dtype=float)),
        where=np.array(b, dtype=float) != 0
    ),
    "v->": lambda a, b: np.array(a, dtype=float) > np.array(b, dtype=float),
    "v-<": lambda a, b: np.array(a, dtype=float) < np.array(b, dtype=float)
}

# 定义标量函数，包含 classify.py 里所有补全的函数，确保训练和推理一致
scalar_functions = {
    "s-median": lambda x: np.median(x) if len(x) > 0 and np.any(x) else 0.0,
    "s-mean": lambda x: np.mean(x) if len(x) > 0 and np.any(x) else 0.0,
    "s-std": lambda x: np.std(x) if len(x) > 1 and np.any(x) else 0.0,
    "s-max": lambda x: np.max(x) if len(x) > 0 and np.any(x) else 0.0,
    "s-min": lambda x: np.min(x) if len(x) > 0 and np.any(x) else 0.0,
    "s-l2-norm": lambda x: np.linalg.norm(x) / len(x) ** 0.5 if len(x) > 0 and np.any(x) else 0.0,
    # classify.py 里补全的函数
    "s-var": lambda x: np.var(x, ddof=1) if len(x) > 1 and np.any(x) else 0.0,
    "s-avg-top-25": lambda x: (
        np.mean(sorted(x, reverse=True)[:25]) if len(x) >= 25 else (
            np.mean(x) if len(x) > 0 else 0.0
        )
    ),
    "s-avg": lambda x: np.mean(x) if len(x) > 0 else 0.0,
    "s-l2": lambda x: np.linalg.norm(x) if len(x) > 0 else 0.0
}

# 定义可用向量
vectors = ["neo-logprobs", "j6b-logprobs", "trigram-logprobs", "unigram-logprobs"]

# 生成向量组合
vec_combinations = defaultdict(list)
for vec1 in range(len(vectors)):
    for vec2 in range(vec1):
        for func in vec_functions:
            if func != "v-div":
                vec_combinations[vectors[vec1]].append(f"{func} {vectors[vec2]}")
for vec1 in vectors:
    for vec2 in vectors:
        if vec1 != vec2:
            vec_combinations[vec1].append(f"v-div {vec2}")

# 全局变量用于 multiprocessing
_file_names = None
_vector_map = None


def convert_file_to_logprob_file(file_path, model_name):
    """
    支持递归向上查找所有父目录下的 logprobs 文件夹，优先返回实际存在的文件。
    """
    base_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    logprob_file_name = f"{file_name_without_ext}-{model_name}.txt"
    # 递归向上查找 logprobs 目录
    cur_dir = base_dir
    while True:
        candidate = os.path.join(cur_dir, "logprobs", logprob_file_name)
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(cur_dir)
        if parent == cur_dir:
            break
        cur_dir = parent
    # 兼容 reuter 结构：如 ghostbuster-data/reuter/gpt/某子文件夹/logprobs
    for root, dirs, files in os.walk(base_dir):
        if "logprobs" in dirs:
            candidate = os.path.join(root, "logprobs", logprob_file_name)
            if os.path.exists(candidate):
                return candidate
    # 默认返回原有规则
    logprob_dir = os.path.join(base_dir, "logprobs")
    os.makedirs(logprob_dir, exist_ok=True)
    return os.path.join(logprob_dir, logprob_file_name)


def get_words(exp):
    return exp.split(" ")


def backtrack_functions(vectors=vectors, max_depth=2):
    def helper(prev, depth):
        if depth >= max_depth:
            return []
        all_funcs = []
        prev_word = get_words(prev)[-1]
        for func in scalar_functions:
            all_funcs.append(f"{prev} {func}")
        for comb in vec_combinations[prev_word]:
            all_funcs += helper(f"{prev} {comb}", depth + 1)
        return all_funcs

    ret = []
    for vec in vectors:
        ret += helper(vec, 0)
    return ret


def train_trigram(verbose=True, return_tokenizer=False):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    sentences = brown.sents()
    if verbose:
        print("Tokenizing corpus...")
    tokenized_corpus = []
    for sentence in tqdm.tqdm(sentences):
        tokens = tokenizer(" ".join(sentence))
        tokenized_corpus += tokens
    if verbose:
        print("\nTraining n-gram model...")
    if return_tokenizer:
        return TrigramBackoff(tokenized_corpus), tokenizer
    else:
        return TrigramBackoff(tokenized_corpus)


def compute_batch_token_logprobs(texts: list, model, tokenizer, batch_size=8, max_length=1024):
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            valid_texts = [t for t in batch_texts if t.strip()]
            if not valid_texts:
                results.extend([([], np.array([]))] * len(batch_texts))
                continue
            # 高效写法：直接 batch encode，设置 truncation 和 padding
            inputs = tokenizer(
                valid_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding="max_length"
            ).to(device)
            assert inputs["input_ids"].shape[1] <= max_length, f"Batch input shape {inputs['input_ids'].shape} > max_length {max_length}"
            if inputs["input_ids"].numel() == 0:
                print(f"Warning: Empty tokenized batch at index {i}")
                results.extend([([], np.array([]))] * len(valid_texts))
                results.extend([([], np.array([]))] * (len(batch_texts) - len(valid_texts)))
                continue
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                logits = outputs.logits
            logprobs = torch.log_softmax(logits, dim=-1).cpu().numpy()
            token_ids = inputs["input_ids"].cpu().numpy()
            for j, tids in enumerate(token_ids):
                valid_tids = tids[tids != tokenizer.pad_token_id]
                if len(valid_tids) == 0:
                    results.append(([], np.array([])))
                    continue
                tokens = tokenizer.convert_ids_to_tokens(valid_tids)
                probs = np.array(
                    [logprobs[j, k, tid] for k, tid in enumerate(valid_tids)],
                    dtype=np.float64
                )
                results.append((tokens, probs))
            results.extend([([], np.array([]))] * (len(batch_texts) - len(valid_texts)))
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            results.extend([([], np.array([]))] * len(batch_texts))
    return results


def get_all_logprobs(
        generate_dataset,
        preprocess=lambda x: x.strip(),
        verbose=True,
        trigram=None,
        tokenizer=None,
        num_tokens=2047,
        batch_size=8
):
    if trigram is None:
        trigram, tokenizer = train_trigram(verbose=verbose, return_tokenizer=True)
    neo_logprobs, j6b_logprobs = {}, {}
    trigram_logprobs, unigram_logprobs = {}, {}
    neo_tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    neo_mdl = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    gpt2_mdl = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

    if neo_tok.pad_token is None:
        neo_tok.pad_token = neo_tok.eos_token
        if verbose:
            print(f"Set neo_tok.pad_token to {neo_tok.pad_token}")
    if gpt2_tok.pad_token is None:
        gpt2_tok.pad_token = gpt2_tok.eos_token
        if verbose:
            print(f"Set gpt2_tok.pad_token to {gpt2_tok.pad_token}")

    if verbose:
        print("Loading logprobs into memory")
    file_names = generate_dataset(lambda file: file, verbose=False)
    if verbose:
        print(f"Total files: {len(file_names)}")

    file_contents = []
    invalid_files = []
    for file in tqdm.tqdm(file_names, desc="Reading files"):
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                doc = preprocess(f.read())
            file_contents.append(doc)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            file_contents.append("")
            invalid_files.append(file)

    to_iter = tqdm.tqdm(range(0, len(file_names), batch_size), desc="Processing batches") if verbose else range(0,
                                                                                                                len(file_names),
                                                                                                                batch_size)
    for batch_idx in to_iter:
        batch_files = file_names[batch_idx:batch_idx + batch_size]
        batch_docs = file_contents[batch_idx:batch_idx + batch_size]
        valid_indices = [i for i, doc in enumerate(batch_docs) if doc.strip()]
        if not valid_indices:
            for file in batch_files:
                print(f"Skipping empty document: {file}")
                invalid_files.append(file)
                neo_logprobs[file] = np.zeros(num_tokens)
                j6b_logprobs[file] = np.zeros(num_tokens)
                trigram_logprobs[file] = np.zeros(num_tokens)
                unigram_logprobs[file] = np.zeros(num_tokens)
                # 保存空 logprob 文件（trigram/unigram 仍可用 np.savetxt）
                neo_file = convert_file_to_logprob_file(file, "neo")
                j6b_file = convert_file_to_logprob_file(file, "j6b")
                # 不再用 np.savetxt 写 neo/j6b
                continue
            continue
        # 关键：分别传 max_length
        neo_results = compute_batch_token_logprobs(
            [batch_docs[i] for i in valid_indices], neo_mdl, neo_tok, batch_size, max_length=2048
        )
        j6b_results = compute_batch_token_logprobs(
            [batch_docs[i] for i in valid_indices], gpt2_mdl, gpt2_tok, batch_size, max_length=1024
        )
        tri_scores_list = []
        uni_scores_list = []
        for i, doc in enumerate(batch_docs):
            if i not in valid_indices:
                tri_scores_list.append(np.zeros(num_tokens))
                uni_scores_list.append(np.zeros(num_tokens))
                continue
            try:
                # 修正：tokenizer.encode
                tri_scores = score_ngram(doc, trigram, tokenizer.encode, n=3, strip_first=False)[:num_tokens]
                uni_scores = score_ngram(doc, trigram.base, tokenizer.encode, n=1, strip_first=False)[:num_tokens]
            except Exception as e:
                print(f"Error in score_ngram for file {batch_files[i]}: {e}")
                tri_scores = np.zeros(num_tokens)
                uni_scores = np.zeros(num_tokens)
            tri_scores_list.append(tri_scores)
            uni_scores_list.append(uni_scores)
        valid_idx = 0
        for i, file in enumerate(batch_files):
            if i not in valid_indices:
                invalid_files.append(file)
                neo_logprobs[file] = np.zeros(num_tokens)
                j6b_logprobs[file] = np.zeros(num_tokens)
                trigram_logprobs[file] = np.zeros(num_tokens)
                unigram_logprobs[file] = np.zeros(num_tokens)
                # 保存空 logprob 文件（trigram/unigram 仍可用 np.savetxt）
                neo_file = convert_file_to_logprob_file(file, "neo")
                j6b_file = convert_file_to_logprob_file(file, "j6b")
                # 不再用 np.savetxt 写 neo/j6b
                continue
            neo_tokens, neo_lp = neo_results[valid_idx]
            j6b_tokens, j6b_lp = j6b_results[valid_idx]
            tri_scores = tri_scores_list[i]
            uni_scores = uni_scores_list[i]
            valid_idx += 1
            # 修正：先截断再 pad，防止负数
            def pad_or_trunc(arr):
                arr = arr[:num_tokens] if len(arr) > num_tokens else arr
                return np.pad(arr, (0, num_tokens - len(arr)), mode='constant', constant_values=0.0) if len(arr) < num_tokens else arr
            neo_lp = pad_or_trunc(neo_lp) if len(neo_lp) > 0 else np.zeros(num_tokens)
            j6b_lp = pad_or_trunc(j6b_lp) if len(j6b_lp) > 0 else np.zeros(num_tokens)
            tri_scores = pad_or_trunc(tri_scores) if len(tri_scores) > 0 else np.zeros(num_tokens)
            uni_scores = pad_or_trunc(uni_scores) if len(uni_scores) > 0 else np.zeros(num_tokens)
            neo_logprobs[file] = neo_lp
            j6b_logprobs[file] = j6b_lp
            trigram_logprobs[file] = tri_scores
            unigram_logprobs[file] = uni_scores
            # 保存 logprob 文件
            neo_file = convert_file_to_logprob_file(file, "neo")
            j6b_file = convert_file_to_logprob_file(file, "j6b")
            os.makedirs(os.path.dirname(neo_file), exist_ok=True)
            os.makedirs(os.path.dirname(j6b_file), exist_ok=True)
            # 只用 write_logprobs 写 neo/j6b
            write_logprobs(batch_docs[i], neo_file, neo_mdl, neo_tok)
            write_logprobs(batch_docs[i], j6b_file, gpt2_mdl, gpt2_tok)
            # trigram/unigram 仍可用 np.savetxt
            # np.savetxt(neo_file, neo_logprobs[file], fmt="%.6f")  # 删除此行
            # np.savetxt(j6b_file, j6b_logprobs[file], fmt="%.6f")  # 删除此行
    if verbose:
        print(f"Completed processing {len(file_names)} files in get_all_logprobs")
        if invalid_files:
            print(f"Invalid files ({len(invalid_files)}): {invalid_files[:5]}...")

    # 验证 logprobs 形状
    for file in file_names:
        for logprobs, name in [
            (neo_logprobs, "neo_logprobs"),
            (j6b_logprobs, "j6b_logprobs"),
            (trigram_logprobs, "trigram_logprobs"),
            (unigram_logprobs, "unigram_logprobs")
        ]:
            lp = logprobs.get(file, np.zeros(num_tokens))
            if not isinstance(lp, np.ndarray) or lp.ndim != 1 or len(lp) != num_tokens:
                print(
                    f"Warning: Invalid {name}[{file}]: shape {lp.shape if isinstance(lp, np.ndarray) else 'N/A'}, type {type(lp)}")
                logprobs[file] = np.zeros(num_tokens)
                # 保存空 logprob 文件
                if name in ["neo_logprobs", "j6b_logprobs"]:
                    model_name = "neo" if name == "neo_logprobs" else "j6b"
                    logprob_file = convert_file_to_logprob_file(file, model_name)
                    os.makedirs(os.path.dirname(logprob_file), exist_ok=True)
                    # np.savetxt(logprob_file, lp, fmt="%.6f")  # 删除此行
    return neo_logprobs, j6b_logprobs, trigram_logprobs, unigram_logprobs


def calc_features_batch(files, exp):
    global _vector_map
    try:
        exp_tokens = get_words(exp)
        if not exp_tokens or exp_tokens[0] not in _vector_map:
            print(f"Invalid feature expression {exp}: missing key {exp_tokens[0]}")
            return np.zeros(len(files))
        # 预处理 logprobs，填充到最大长度
        logprobs_list = []
        max_len = 0
        for f in files:
            lp = _vector_map.get(exp_tokens[0], lambda x: np.zeros(2047))(f)
            if isinstance(lp, np.ndarray) and lp.size > 0 and lp.ndim == 1:
                logprobs_list.append(lp)
                max_len = max(max_len, len(lp))
            else:
                logprobs_list.append(np.zeros(2047))
        if not any(len(lp) > 0 and np.any(lp) for lp in logprobs_list):
            return np.zeros(len(files))
        # 填充到 max_len
        padded_logprobs = []
        for lp in logprobs_list:
            if len(lp) == 0 or not np.any(lp):
                padded_logprobs.append(np.zeros(max_len))
            else:
                padded = np.pad(lp, (0, max_len - len(lp)), mode='constant', constant_values=0.0)
                padded_logprobs.append(padded)
        curr = np.array(padded_logprobs)
        valid = np.array([len(lp) > 0 and np.any(lp) for lp in logprobs_list])
        if not np.any(valid):
            return np.zeros(len(files))
        curr = curr[valid]
        for i in range(1, len(exp_tokens)):
            if exp_tokens[i] in vec_functions:
                next_key = exp_tokens[i + 1]
                if next_key not in _vector_map:
                    print(f"Invalid feature expression {exp}: missing key {next_key}")
                    return np.zeros(len(files))
                next_vec_list = []
                for f in files:
                    lp = _vector_map.get(next_key, lambda x: np.zeros(2047))(f)
                    if isinstance(lp, np.ndarray) and lp.size > 0 and lp.ndim == 1:
                        padded = np.pad(lp, (0, max_len - len(lp)), mode='constant', constant_values=0.0)
                        next_vec_list.append(padded)
                    else:
                        next_vec_list.append(np.zeros(max_len))
                next_vec = np.array(next_vec_list)[valid]
                valid_next = np.array([len(v) > 0 and np.any(v) for v in next_vec])
                if not np.any(valid_next):
                    return np.zeros(len(files))
                curr = curr[valid_next]
                next_vec = next_vec[valid_next]
                curr = vec_functions[exp_tokens[i]](curr, next_vec)
                valid[valid] = valid_next
            elif exp_tokens[i] in scalar_functions:
                results = np.zeros(len(files))
                results[valid] = [scalar_functions[exp_tokens[i]](c) for c in curr]
                return results
        return curr
    except Exception as e:
        print(f"Error in calc_features_batch for {exp}: {e}")
        return np.zeros(len(files))


def calc_features_for_exp(exp):
    global _file_names
    try:
        data = calc_features_batch(_file_names, exp).reshape(-1, 1)
        return exp, data
    except Exception as e:
        print(f"Error computing feature {exp}: {e}")
        return exp, np.zeros((len(_file_names), 1))


def generate_symbolic_data(
        generate_dataset,
        preprocess=lambda x: x.strip(),
        max_depth=2,
        output_file="symbolic_data",
        verbose=True,
        vector_map_keys=None,
        num_processes=4
):
    global _file_names, _vector_map
    if vector_map_keys is None:
        vector_map_keys = ["neo-logprobs", "j6b-logprobs", "trigram-logprobs", "unigram-logprobs"]
    global vectors
    vectors = vector_map_keys
    global vec_combinations
    vec_combinations = defaultdict(list)
    for vec1 in range(len(vectors)):
        for vec2 in range(vec1):
            for func in vec_functions:
                if func != "v-div":
                    vec_combinations[vectors[vec1]].append(f"{func} {vectors[vec2]}")
    for vec1 in vectors:
        for vec2 in vectors:
            if vec1 != vec2:
                vec_combinations[vec1].append(f"v-div {vec2}")
    neo_logprobs, j6b_logprobs, trigram_logprobs, unigram_logprobs = get_all_logprobs(
        generate_dataset, preprocess=preprocess, verbose=verbose, batch_size=8
    )
    _file_names = generate_dataset(lambda file: file, verbose=False)
    _vector_map = {
        "neo-logprobs": lambda file: neo_logprobs.get(file, np.zeros(2047)),
        "j6b-logprobs": lambda file: j6b_logprobs.get(file, np.zeros(2047)),
        "trigram-logprobs": lambda file: trigram_logprobs.get(file, np.zeros(2047)),
        "unigram-logprobs": lambda file: unigram_logprobs.get(file, np.zeros(2047))
    }
    all_funcs = backtrack_functions(vectors=vectors, max_depth=max_depth)
    if verbose:
        print(f"\nTotal # of Features: {len(all_funcs)}.")
        print("Sampling 5 features:")
        for i in range(5):
            print(all_funcs[np.random.randint(0, len(all_funcs))])
        print("\nGenerating datasets...")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    exp_to_data = {}
    with Pool(processes=num_processes) as pool:
        for exp, data in tqdm.tqdm(
                pool.imap_unordered(calc_features_for_exp, all_funcs),
                total=len(all_funcs)
        ):
            exp_to_data[exp] = data
    pickle.dump(exp_to_data, open(output_file, "wb"))


def get_exp_featurize(best_features, vector_map):
    import numpy as np
    def calc_features(file_or_vec, exp):
        exp_tokens = get_words(exp)
        try:
            curr = vector_map[exp_tokens[0]](file_or_vec) if callable(vector_map[exp_tokens[0]]) else vector_map[exp_tokens[0]]
            curr = np.array(curr, dtype=float)
            for i in range(1, len(exp_tokens)):
                if exp_tokens[i] in vec_functions:
                    next_vec = vector_map[exp_tokens[i + 1]](file_or_vec) if callable(vector_map[exp_tokens[i + 1]]) else vector_map[exp_tokens[i + 1]]
                    next_vec = np.array(next_vec, dtype=float)
                    curr = vec_functions[exp_tokens[i]](curr, next_vec)
                elif exp_tokens[i] in scalar_functions:
                    val = scalar_functions[exp_tokens[i]](curr)
                    return val
            return curr
        except Exception as e:
            return np.nan
    def exp_featurize(file_or_vec):
        return np.array([calc_features(file_or_vec, exp) for exp in best_features])
    return exp_featurize