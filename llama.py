import os
import argparse
import math
import tqdm
import numpy as np
import dill as pickle
import tiktoken
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.util import ngrams
from nltk.corpus import brown
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from utils.featurize import normalize, score_ngram
from utils.load import Dataset, get_generate_dataset
from utils.n_gram import TrigramBackoff
from utils.symbolic import vec_functions, scalar_functions

# 数据集定义
datasets = [
    Dataset("normal", "ghostbuster-data/wp/human"),
    Dataset("normal", "ghostbuster-data/wp/gpt"),
    Dataset("author", "ghostbuster-data/reuter/human"),
    Dataset("author", "ghostbuster-data/reuter/gpt"),
    Dataset("normal", "ghostbuster-data/essay/human"),
    Dataset("normal", "ghostbuster-data/essay/gpt"),
]

# 使用与 train.py 一致的 best_features（假设已更新）
best_features = [
    "neo-logprobs v-add j6b-logprobs s-avg",
    "trigram-logprobs v-div unigram-logprobs s-avg-top-25",
    "neo-logprobs v-mul j6b-logprobs s-avg",
    "trigram-logprobs v-< unigram-logprobs s-avg",
    "neo-logprobs v-> j6b-logprobs s-var",
    "trigram-logprobs v-mul unigram-logprobs s-min",
    "neo-logprobs v-sub j6b-logprobs s-avg",
    "trigram-logprobs v-div j6b-logprobs s-min",
    "neo-logprobs v-< j6b-logprobs s-avg-top-25",
    "trigram-logprobs v-add unigram-logprobs s-avg",
]

models = ["gpt"]
domains = ["wp", "reuter", "essay"]
eval_domains = ["claude", "gpt_prompt1", "gpt_prompt2", "gpt_writing", "gpt_semantic"]

# 向量定义
vectors = ["neo-logprobs", "j6b-logprobs", "trigram-logprobs", "unigram-logprobs"]

parser = argparse.ArgumentParser()
parser.add_argument("--feature_select", action="store_true")
parser.add_argument("--classify", action="store_true")
args = parser.parse_args()

# 初始化 tokenizer 和 trigram 模型
tokenizer = AutoTokenizer.from_pretrained("gpt2")
sentences = brown.sents()

tokenized_corpus = []
for sentence in tqdm.tqdm(sentences):
    tokens = tokenizer.encode(" ".join(sentence))
    tokenized_corpus += tokens

trigram = TrigramBackoff(tokenized_corpus)

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

def get_words(exp):
    """
    Splits up expression into words, to be individually processed
    """
    return exp.split(" ")

def backtrack_functions(max_depth=2):
    """
    Backtrack all possible features.
    """
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_all_logprobs(
    generate_dataset,
    preprocess=lambda x: x.strip(),
    verbose=True,
    trigram=None,
    tokenizer=None,
    num_tokens=2047,
):
    """
    Computes log probabilities for all files in the dataset using GPT-Neo-125M and GPT2
    """
    if trigram is None:
        trigram = TrigramBackoff(tokenized_corpus)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    neo_logprobs, j6b_logprobs = {}, {}
    trigram_logprobs, unigram_logprobs = {}, {}

    # 加载模型
    neo_tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    neo_mdl = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    gpt2_mdl = AutoModelForCausalLM.from_pretrained("gpt2")

    if verbose:
        print("Loading logprobs into memory")

    file_names = generate_dataset(lambda file: file, verbose=False)
    to_iter = tqdm.tqdm(file_names) if verbose else file_names

    for file in to_iter:
        if "logprobs" in file:
            continue
        with open(file, "r", encoding="utf-8") as f:
            doc = preprocess(f.read())
        if not doc:
            print(f"Skipping empty file: {file}")
            continue
        _, neo_lp = compute_token_logprobs(doc, neo_mdl, neo_tok, max_tokens=2048)
        _, j6b_lp = compute_token_logprobs(doc, gpt2_mdl, gpt2_tok, max_tokens=1024)
        tri_scores = score_ngram(doc, trigram, tokenizer.encode, n=3, strip_first=False)[:num_tokens]
        uni_scores = score_ngram(doc, trigram.base, tokenizer.encode, n=1, strip_first=False)[:num_tokens]
        L = min(len(neo_lp), len(j6b_lp), len(tri_scores), len(uni_scores), num_tokens)
        neo_logprobs[file] = neo_lp[:L]
        j6b_logprobs[file] = j6b_lp[:L]
        trigram_logprobs[file] = tri_scores[:L]
        unigram_logprobs[file] = uni_scores[:L]

    return neo_logprobs, j6b_logprobs, trigram_logprobs, unigram_logprobs

all_funcs = backtrack_functions(max_depth=3)
np.random.seed(0)

# 构造 train/test 分割
indices = np.arange(6000)
np.random.shuffle(indices)
train, test = (
    indices[: math.floor(0.8 * len(indices))],
    indices[math.floor(0.8 * len(indices)) :],
)
print("Train/Test Split:", train, test)

generate_dataset_fn = get_generate_dataset(*datasets)
labels = generate_dataset_fn(
    lambda file: 1 if any([m in file for m in ["gpt", "claude"]]) else 0
)

# 构造所有索引
def get_indices(filter_fn):
    where = np.where(generate_dataset_fn(filter_fn))[0]
    curr_train = [i for i in train if i in where]
    curr_test = [i for i in test if i in where]
    return curr_train, curr_test

indices_dict = {}
for model in models + ["human"]:
    train_indices, test_indices = get_indices(
        lambda file: 1 if model in file else 0,
    )
    indices_dict[f"{model}_train"] = train_indices
    indices_dict[f"{model}_test"] = test_indices

for model in models + ["human"]:
    for domain in domains:
        train_key = f"{model}_{domain}_train"
        test_key = f"{model}_{domain}_test"
        train_indices, test_indices = get_indices(
            lambda file: 1 if domain in file and model in file else 0,
        )
        indices_dict[train_key] = train_indices
        indices_dict[test_key] = test_indices

if args.feature_select:
    neo_logprobs, j6b_logprobs, trigram_logprobs, unigram_logprobs = get_all_logprobs(
        generate_dataset_fn,
        verbose=True,
        tokenizer=tokenizer,
        trigram=trigram,
    )
    vector_map = {
        "neo-logprobs": lambda file: neo_logprobs[file],
        "j6b-logprobs": lambda file: j6b_logprobs[file],
        "trigram-logprobs": lambda file: trigram_logprobs[file],
        "unigram-logprobs": lambda file: unigram_logprobs[file],
    }

    def calc_features(file, exp):
        exp_tokens = get_words(exp)
        curr = vector_map[exp_tokens[0]](file)
        for i in range(1, len(exp_tokens)):
            if exp_tokens[i] in vec_functions:
                next_vec = vector_map[exp_tokens[i + 1]](file)
                curr = vec_functions[exp_tokens[i]](curr, next_vec)
            elif exp_tokens[i] in scalar_functions:
                return scalar_functions[exp_tokens[i]](curr)

    print("Preparing exp_to_data")
    exp_to_data = {}
    for exp in tqdm.tqdm(all_funcs):
        exp_to_data[exp] = generate_dataset_fn(
            lambda file: calc_features(file, exp)
        ).reshape(-1, 1)

    from utils.featurize import select_features
    select_features(exp_to_data, labels, verbose=True, to_normalize=True, indices=train)

if args.classify:
    neo_logprobs, j6b_logprobs, trigram_logprobs, unigram_logprobs = get_all_logprobs(
        generate_dataset_fn,
        verbose=True,
        tokenizer=tokenizer,
        trigram=trigram,
    )
    vector_map = {
        "neo-logprobs": lambda file: neo_logprobs[file],
        "j6b-logprobs": lambda file: j6b_logprobs[file],
        "trigram-logprobs": lambda file: trigram_logprobs[file],
        "unigram-logprobs": lambda file: unigram_logprobs[file],
    }

    def get_exp_featurize(best_features, vector_map):
        def calc_features(file, exp):
            exp_tokens = get_words(exp)
            curr = np.array(vector_map[exp_tokens[0]](file), dtype=float)
            for i in range(1, len(exp_tokens)):
                if exp_tokens[i] in vec_functions:
                    next_vec = np.array(vector_map[exp_tokens[i + 1]](file), dtype=float)
                    curr = vec_functions[exp_tokens[i]](curr, next_vec)
                elif exp_tokens[i] in scalar_functions:
                    return scalar_functions[exp_tokens[i]](curr)
        def exp_featurize(file):
            return np.array([calc_features(file, exp) for exp in best_features])
        return exp_featurize

    data = generate_dataset_fn(get_exp_featurize(best_features, vector_map))
    data = normalize(data)

    def train_model(data, train, test):
        model = LogisticRegression(class_weight='balanced')
        model.fit(data[train], labels[train])
        return f1_score(labels[test], model.predict(data[test]))

    print(
        f"In-Domain: {train_model(data, indices_dict['gpt_train'] + indices_dict['human_train'], indices_dict['gpt_test'] + indices_dict['human_test'])}"
    )

    for test_domain in domains:
        train_indices = []
        for train_domain in domains:
            if train_domain == test_domain:
                continue
            train_indices += (
                indices_dict[f"gpt_{train_domain}_train"]
                + indices_dict[f"human_{train_domain}_train"]
            )
        print(
            f"Out-Domain ({test_domain}): {train_model(data, train_indices, indices_dict[f'gpt_{test_domain}_test'] + indices_dict[f'human_{test_domain}_test'])}"
        )
