import argparse
import math
import numpy as np
import tiktoken
import dill as pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

from tabulate import tabulate

from utils.featurize import normalize, t_featurize, t_featurize_logprobs, score_ngram, select_features
from utils.symbolic import train_trigram, get_exp_featurize, generate_symbolic_data
from utils.load import get_generate_dataset, Dataset

warnings.filterwarnings("ignore")

# 定义数据集
wp_dataset = [
    Dataset("normal", "ghostbuster-data/wp/human"),
    Dataset("normal", "ghostbuster-data/wp/gpt"),
]

reuter_dataset = [
    Dataset("author", "ghostbuster-data/reuter/human"),
    Dataset("author", "ghostbuster-data/reuter/gpt"),
]

essay_dataset = [
    Dataset("normal", "ghostbuster-data/essay/human"),
    Dataset("normal", "ghostbuster-data/essay/gpt"),
]

eval_dataset = [
    Dataset("normal", "ghostbuster-data/wp/claude"),
    Dataset("author", "ghostbuster-data/reuter/claude"),
    Dataset("normal", "ghostbuster-data/essay/claude"),
    Dataset("normal", "ghostbuster-data/wp/gpt_prompt1"),
    Dataset("author", "ghostbuster-data/reuter/gpt_prompt1"),
    Dataset("normal", "ghostbuster-data/essay/gpt_prompt1"),
    Dataset("normal", "ghostbuster-data/wp/gpt_prompt2"),
    Dataset("author", "ghostbuster-data/reuter/gpt_prompt2"),
    Dataset("normal", "ghostbuster-data/essay/gpt_prompt2"),
    Dataset("normal", "ghostbuster-data/wp/gpt_writing"),
    Dataset("author", "ghostbuster-data/reuter/gpt_writing"),
    Dataset("normal", "ghostbuster-data/essay/gpt_writing"),
    Dataset("normal", "ghostbuster-data/wp/gpt_semantic"),
    Dataset("author", "ghostbuster-data/reuter/gpt_semantic"),
    Dataset("normal", "ghostbuster-data/essay/gpt_semantic"),
]


def compute_token_logprobs(text: str, model, tokenizer, max_tokens=1024):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=max_tokens, truncation=True)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        logits = outputs.logits
    logprobs = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    token_ids = inputs.squeeze(0).cpu().numpy().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    probs = np.array([logprobs[i, tid] for i, tid in enumerate(token_ids)], dtype=np.float64)
    return tokens, probs


def get_all_logprobs(generate_dataset_fn, trigram, tokenizer):
    neo_logprobs, gpt2_logprobs, trigram_logprobs, unigram_logprobs = [], [], [], []

    neo_tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    neo_mdl = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    gpt2_mdl = AutoModelForCausalLM.from_pretrained("gpt2")

    files = generate_dataset_fn(lambda x: x)

    for file in files:
        with open(file, encoding="utf-8") as f:
            text = f.read().strip()

        # 计算 logprobs
        _, neo_lp = compute_token_logprobs(text, neo_mdl, neo_tok, max_tokens=2048)
        _, gpt2_lp = compute_token_logprobs(text, gpt2_mdl, gpt2_tok, max_tokens=1024)

        # n-gram 分数
        tri_scores = score_ngram(text, trigram, tokenizer.encode, n=3, strip_first=False)
        uni_scores = score_ngram(text, trigram.base, tokenizer.encode, n=1, strip_first=False)

        # 截断到最小长度
        L = min(len(neo_lp), len(gpt2_lp), len(tri_scores), len(uni_scores))
        neo_logprobs.append(neo_lp[:L])
        gpt2_logprobs.append(gpt2_lp[:L])
        trigram_logprobs.append(tri_scores[:L])
        unigram_logprobs.append(uni_scores[:L])

    return neo_logprobs, gpt2_logprobs, trigram_logprobs, unigram_logprobs


def get_featurized_data(generate_dataset_fn, best_features):
    try:
        t_data = generate_dataset_fn(t_featurize)
    except Exception as e:
        return None

    neo_logprobs, gpt2_logprobs, trigram_logprobs, unigram_logprobs = get_all_logprobs(
        generate_dataset_fn, trigram=trigram_model, tokenizer=tokenizer
    )

    files = generate_dataset_fn(lambda x: x)
    if not isinstance(files, list):
        try:
            files = list(files)
        except Exception as e:
            files = []
    file_to_idx = {file: idx for idx, file in enumerate(files)}
    vector_map = {
        "neo-logprobs": lambda file: neo_logprobs[file_to_idx[file]],
        "j6b-logprobs": lambda file: gpt2_logprobs[file_to_idx[file]],
        "trigram-logprobs": lambda file: trigram_logprobs[file_to_idx[file]],
        "unigram-logprobs": lambda file: unigram_logprobs[file_to_idx[file]]
    }
    exp_featurize = get_exp_featurize(best_features, vector_map)
    exp_data_list = []
    for i, file in enumerate(files):
        try:
            features = exp_featurize(file)
            exp_data_list.append(features)
        except Exception as e:
            pass
    if len(exp_data_list) == 0:
        return np.empty((0,))
    exp_data = np.array(exp_data_list)
    try:
        concat = np.concatenate([t_data, exp_data], axis=1)
    except Exception as e:
        return np.empty((0,))
    return concat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_symbolic_data", action="store_true")
    parser.add_argument("--generate_symbolic_data_four", action="store_true")
    parser.add_argument("--generate_symbolic_data_eval", action="store_true")
    parser.add_argument("--perform_feature_selection", action="store_true")
    parser.add_argument("--perform_feature_selection_one", action="store_true")
    parser.add_argument("--perform_feature_selection_two", action="store_true")
    parser.add_argument("--perform_feature_selection_four", action="store_true")
    parser.add_argument("--perform_feature_selection_only_ada", action="store_true")
    parser.add_argument("--perform_feature_selection_no_gpt", action="store_true")
    parser.add_argument("--perform_feature_selection_domain", action="store_true")
    parser.add_argument("--only_include_gpt", action="store_true")
    parser.add_argument("--train_on_all_data", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(">>> train.py main started, args:", args)

    np.random.seed(args.seed)

    result_table = [["F1", "Accuracy", "AUC"]]

    datasets = [
        *wp_dataset,
        *reuter_dataset,
        *essay_dataset,
    ]
    generate_dataset_fn = get_generate_dataset(*datasets)

    print("Loading trigram model...")
    trigram_model = pickle.load(open("model/trigram_model.pkl", "rb"))
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if args.generate_symbolic_data:
        generate_symbolic_data(
            generate_dataset_fn,
            max_depth=3,
            output_file="symbolic_data_gpt",
            verbose=True,
            vector_map_keys=["neo-logprobs", "j6b-logprobs", "trigram-logprobs", "unigram-logprobs"]
        )
        t_data = generate_dataset_fn(t_featurize)
        pickle.dump(t_data, open("t_data", "wb"))

    if args.generate_symbolic_data_eval:
        generate_dataset_fn_eval = get_generate_dataset(*eval_dataset)
        generate_symbolic_data(
            generate_dataset_fn_eval,
            max_depth=3,
            output_file="symbolic_data_eval",
            verbose=True,
            vector_map_keys=["neo-logprobs", "j6b-logprobs", "trigram-logprobs", "unigram-logprobs"]
        )
        t_data_eval = generate_dataset_fn_eval(t_featurize)
        pickle.dump(t_data_eval, open("t_data_eval", "wb"))

    if args.generate_symbolic_data_four:
        generate_symbolic_data(
            generate_dataset_fn,
            max_depth=4,
            output_file="symbolic_data_gpt_four",
            verbose=True,
            vector_map_keys=["neo-logprobs", "j6b-logprobs", "trigram-logprobs", "unigram-logprobs"]
        )
        t_data = generate_dataset_fn(t_featurize)
        pickle.dump(t_data, open("t_data", "wb"))

    labels = generate_dataset_fn(
        lambda file: 1 if any([m in file for m in ["gpt", "claude"]]) else 0
    )

    indices = np.arange(len(labels))
    if args.only_include_gpt:
        where_gpt = np.where(
            generate_dataset_fn(lambda file: 0 if "claude" in file else 1)
        )[0]
        indices = indices[where_gpt]

    np.random.shuffle(indices)
    train, test = (
        indices[: math.floor(0.8 * len(indices))],
        indices[math.floor(0.8 * len(indices)):],
    )
    print("Train/Test Split", train, test)
    print("Train Size:", len(train), "Valid Size:", len(test))
    print(f"Positive Labels: {sum(labels[indices])}, Total Labels: {len(indices)}")

    if args.perform_feature_selection:
        print(">>> Entering perform_feature_selection branch")
        exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))
        best_features = select_features(
            exp_to_data, labels, verbose=True, to_normalize=True, indices=train, max_features=9
        )
        with open("results/best_features_three.txt", "w") as f:
            for feat in best_features:
                f.write(feat + "\n")

    if args.perform_feature_selection_two:
        old_exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))
        exp_to_data = {}
        for key in old_exp_to_data:
            if len(key.split(" ")) <= 4:
                exp_to_data[key] = old_exp_to_data[key]
        best_features = select_features(
            exp_to_data, labels, verbose=True, to_normalize=True, indices=train, max_features=9
        )
        with open("results/best_features_two.txt", "w") as f:
            for feat in best_features:
                f.write(feat + "\n")

    if args.perform_feature_selection_one:
        old_exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))
        exp_to_data = {}
        for key in old_exp_to_data:
            if len(key.split(" ")) <= 2:
                exp_to_data[key] = old_exp_to_data[key]
        best_features = select_features(
            exp_to_data, labels, verbose=True, to_normalize=True, indices=train, max_features=9
        )
        with open("results/best_features_one.txt", "w") as f:
            for feat in best_features:
                f.write(feat + "\n")

    if args.perform_feature_selection_four:
        exp_to_data = pickle.load(open("symbolic_data_gpt_four", "rb"))
        best_features = select_features(
            exp_to_data, labels, verbose=True, to_normalize=True, indices=train, max_features=9
        )
        with open("results/best_features_four.txt", "w") as f:
            for feat in best_features:
                f.write(feat + "\n")

    if args.perform_feature_selection_no_gpt:
        old_exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))
        exp_to_data = {}
        for key in old_exp_to_data:
            if "neo" not in key and "j6b" not in key:
                exp_to_data[key] = old_exp_to_data[key]
        best_features = select_features(
            exp_to_data, labels, verbose=True, to_normalize=True, indices=train, max_features=9
        )
        with open("results/best_features_no_gpt.txt", "w") as f:
            for feat in best_features:
                f.write(feat + "\n")

    if args.perform_feature_selection_only_ada:
        old_exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))
        exp_to_data = {}
        for key in old_exp_to_data:
            if "neo" not in key:
                exp_to_data[key] = old_exp_to_data[key]
        best_features = select_features(
            exp_to_data, labels, verbose=True, to_normalize=True, indices=train, max_features=9
        )
        with open("results/best_features_only_ada.txt", "w") as f:
            for feat in best_features:
                f.write(feat + "\n")

    if args.perform_feature_selection_domain:
        exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))
        wp_indices = np.where(generate_dataset_fn(lambda file: "wp" in file))[0]
        reuter_indices = np.where(generate_dataset_fn(lambda file: "reuter" in file))[0]
        essay_indices = np.where(generate_dataset_fn(lambda file: "essay" in file))[0]

        wp_features = select_features(
            exp_to_data, labels, verbose=True, to_normalize=True, indices=wp_indices, max_features=9
        )
        with open("results/best_features_wp.txt", "w") as f:
            for feat in wp_features:
                f.write(feat + "\n")

        reuter_features = select_features(
            exp_to_data, labels, verbose=True, to_normalize=True, indices=reuter_indices, max_features=9
        )
        with open("results/best_features_reuter.txt", "w") as f:
            for feat in reuter_features:
                f.write(feat + "\n")

        essay_features = select_features(
            exp_to_data, labels, verbose=True, to_normalize=True, indices=essay_indices, max_features=9
        )
        with open("results/best_features_essay.txt", "w") as f:
            for feat in essay_features:
                f.write(feat + "\n")

    if args.train_on_all_data:
        # 用train.py的实现替换，保证一致性
        t_data = pickle.load(open("t_data", "rb"))
        exp_dict = pickle.load(open("symbolic_data_gpt", "rb"))
        labels = np.load("labels.npy", allow_pickle=True)

        # 加载 best_features_three.txt
        with open("results/best_features_three.txt", "r", encoding="utf-8") as f:
            best_features = [line.strip() for line in f if line.strip()]

        # 只拼接 best_features 里的特征
        exp_data = np.concatenate([
            np.array(exp_dict[k]).reshape(-1, 1) for k in best_features if k in exp_dict
        ], axis=1)
        data = np.concatenate([t_data, exp_data], axis=1)

        # 自动过滤掉所有包含 NaN/Inf 的特征列，并记录有效特征名
        t_feat_names = [f"t_feat_{i}" for i in range(t_data.shape[1])]
        all_feat_names = t_feat_names + best_features  # 保持原有特征名
        valid_cols = []
        valid_feat_names = []
        for i in range(data.shape[1]):
            col = data[:, i]
            if np.any(np.isnan(col)) or np.any(np.isinf(col)):
                continue
            valid_cols.append(i)
            valid_feat_names.append(all_feat_names[i])
        data = data[:, valid_cols]

        # 保存有效特征名
        os.makedirs("model", exist_ok=True)
        with open("model/features.txt", "w", encoding="utf-8") as f:
            for feat in valid_feat_names:
                f.write(feat + "\n")
        with open("model/all_feat_names.txt", "w", encoding="utf-8") as f:
            for name in all_feat_names:
                f.write(name + "\n")

        # 计算并保存 mu 和 sigma
        mu = data.mean(axis=0)
        sigma = data.std(axis=0)
        pickle.dump(mu, open("model/mu", "wb"))
        pickle.dump(sigma, open("model/sigma", "wb"))

        # 训练/评估
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        train, test = (
            indices[: int(0.8 * len(indices))],
            indices[int(0.8 * len(indices)) :],
        )
        print("Train/Test Split", train, test)
        print("Train Size:", len(train), "Valid Size:", len(test))
        print(f"Positive Labels: {sum(labels[indices])}, Total Labels: {len(indices)}")

        base = LogisticRegression()
        model = CalibratedClassifierCV(base, cv=5)
        model.fit(data[train], labels[train])
        pickle.dump(model, open("model/model", "wb"))

        predictions = model.predict(data[test])
        probs = model.predict_proba(data[test])[:, 1]
        result_table = [
            ["F1", "Accuracy", "AUC"],
            [
                round(f1_score(labels[test], predictions), 3),
                round(accuracy_score(labels[test], predictions), 3),
                round(roc_auc_score(labels[test], probs), 3),
            ],
        ]
        print(tabulate(result_table, headers="firstrow", tablefmt="grid"))
        print("模型训练与保存完成。")
        exit(0)

    # 加载 symbolic_data_gpt
    with open("symbolic_data_gpt", "rb") as f:
        exp_dict = pickle.load(f)
    exp_keys = [k for k in all_feat_names if k.startswith("exp_") or k in exp_dict]  # 你实际的 key 规则

    # 对当前输入文本，生成 exp_feats
    exp_feats = []
    for k in exp_keys:
        # 这里应调用和训练时一致的 featurize 逻辑
        val = your_exp_featurize_function(input_text, k)  # 你实际的特征生成函数
        exp_feats.append(val)
