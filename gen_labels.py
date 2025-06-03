# 生成 labels.npy 的脚本
import numpy as np
from utils.load import get_generate_dataset, Dataset

datasets = [
    Dataset("normal", "ghostbuster-data/wp/human"),
    Dataset("normal", "ghostbuster-data/wp/gpt"),
    Dataset("author", "ghostbuster-data/reuter/human"),
    Dataset("author", "ghostbuster-data/reuter/gpt"),
    Dataset("normal", "ghostbuster-data/essay/human"),
    Dataset("normal", "ghostbuster-data/essay/gpt"),
]
generate_dataset_fn = get_generate_dataset(*datasets)
files = generate_dataset_fn(lambda x: x)
labels = np.array([1 if ("gpt" in f or "claude" in f) and not "human" in f else 0 for f in files])
np.save("labels.npy", labels)
print(f"标签已生成，正例数: {labels.sum()}，总样本数: {len(labels)}")
