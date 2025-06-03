import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import argparse
import re
import tqdm
import os
import math
import nltk
import numpy as np
import string
import torch
from nltk.corpus import wordnet
from datasets import load_dataset
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.generate import generate_documents
from utils.write_logprobs import write_logprobs
from utils.symbolic import convert_file_to_logprob_file
from utils.load import Dataset, get_generate_dataset

nltk.download("wordnet")
nltk.download("omw-1.4")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化 tokenizer 和模型
neo_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
neo_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M", device_map=None)
neo_model = neo_model.to(device)
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2", device_map=None)
gpt2_model = gpt2_model.to(device)

datasets = [
    Dataset("normal", "ghostbuster-data/wp/human"),
    Dataset("normal", "ghostbuster-data/wp/gpt"),
    Dataset("author", "ghostbuster-data/reuter/human"),
    Dataset("author", "ghostbuster-data/reuter/gpt"),
    Dataset("normal", "ghostbuster-data/essay/human"),
    Dataset("normal", "ghostbuster-data/essay/gpt"),
]
generate_dataset_fn = get_generate_dataset(*datasets)

prompt_types = ["gpt", "gpt_prompt1", "gpt_prompt2", "gpt_writing", "gpt_semantic"]
html_replacements = [
    ("&", "&"),
    ("<", "<"),
    (">", ">"),
    ('"', '"'),
    ("'", "'"),
]

perturb_char_names = [
    "char_basic",
    "char_space",
    "char_cap",
    "word_adj",
    "word_syn",
]
perturb_char_sizes = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200]

perturb_sent_names = ["sent_adj", "sent_paraph", "para_adj", "para_paraph"]
perturb_sent_sizes = list(range(11))

def closest_synonym(word):
    synonyms = wordnet.synsets(word)
    if not synonyms:
        return None
    closest_synset = synonyms[0]
    for synset in synonyms[1:]:
        if len(synset.lemmas()) > len(closest_synset.lemmas()):
            closest_synset = synset
    for lemma in closest_synset.lemmas():
        if lemma.name() != word:
            return lemma.name()
    return None

def html_replace(text):
    for replacement in html_replacements:
        text = text.replace(replacement[0], replacement[1])
    return text

def round_to_100(n):
    return int(round(n / 100.0)) * 100

def get_wp_prompts(words, prompt):
    return [
        f'Write a story in {words} words to the prompt "{prompt}".',
        f'You are an author, who is writing a story in response to the prompt "{prompt}". What would you write in {words} words?',
        f'Hi! I\'m trying to write a {words}-word story on the following prompt: "{prompt}". Could you please draft something for me?',
        f'Please help me write a short story in response to the prompt "{prompt}".',
        f'Write a {words}-word story in the style of a beginner writer in response to the prompt "{prompt}".',
        f'Write a story with very short sentences in {words} words to the prompt "{prompt}".',
    ]

def get_reuter_prompts(words, headline):
    return [
        f'Write a news article in {words} words based on the headline "{headline}".',
        f'You are a news reporter, who is writing an article with the headline "{headline}". What would you write in {words} words?',
        f'Hi! I\'m trying to write a {words}-word news article based on the following headline: "{headline}". Could you please draft something for me?',
        f'Please help me write a New York Times article for the headline "{headline}".',
        f'Write a {words}-word news article in the style of a New York Times article based on the headline "{headline}".',
        f'Write a news article with very short sentences in {words} words based on the headline "{headline}".',
    ]

def get_essay_prompts(words, prompt):
    return [
        f'Write an essay in {words} words to the prompt "{prompt}".',
        f'You are a student, who is writing an essay in response to the prompt "{prompt}". What would you write in {words} words?',
        f'Hi! I\'m trying to write a {words}-word essay based on the following prompt: "{prompt}". Could you please draft something for me?',
        f'Please help me write an essay in response to the prompt "{prompt}".',
        f'Write a {words}-word essay in the style of a high-school student in response to the following prompt: "{prompt}".',
        f'Write an essay with very short sentences in {words} words to the prompt "{prompt}".',
    ]

def generate_logprobs(generate_dataset_fn):
    files = generate_dataset_fn(lambda f: f)
    for file in tqdm.tqdm(files):
        if "logprobs" in file:
            continue
        base_path = os.path.dirname(file) + "/logprobs"
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
        try:
            with open(file, "r", encoding="utf-8") as f:
                doc = f.read().strip()
            if not doc or all(c in " \n\t\r" for c in doc):
                print(f"[Warning] Skipping file {file} due to empty or whitespace-only content.")
                continue

            # 直接传原始 doc，write_logprobs 内部会严格截断
            neo_file = convert_file_to_logprob_file(file, "neo")
            if not os.path.exists(neo_file):
                write_logprobs(doc, neo_file, neo_model, neo_tokenizer)
            gpt2_file = convert_file_to_logprob_file(file, "gpt2")
            if not os.path.exists(gpt2_file):
                write_logprobs(doc, gpt2_file, gpt2_model, gpt2_tokenizer)

        except Exception as e:
            print(f"[Warning] Skipping file {file} due to error: {e}")

def compute_token_logprobs(text: str, model, tokenizer, max_tokens=1024):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wp_prompts", action="store_true")
    parser.add_argument("--wp_human", action="store_true")
    parser.add_argument("--wp_gpt", action="store_true")
    parser.add_argument("--reuter_human", action="store_true")
    parser.add_argument("--reuter_gpt", action="store_true")
    parser.add_argument("--essay_prompts", action="store_true")
    parser.add_argument("--essay_human", action="store_true")
    parser.add_argument("--essay_gpt", action="store_true")
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--logprob_other", action="store_true")
    parser.add_argument("--gen_perturb_char", action="store_true")
    parser.add_argument("--logprob_perturb_char", action="store_true")
    parser.add_argument("--gen_perturb_sent", action="store_true")
    parser.add_argument("--logprob_perturb_sent", action="store_true")
    parser.add_argument("--batch_start", type=int, default=0, help="Batch start index")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size")
    args = parser.parse_args()

    if args.wp_prompts:
        def format_prompt(p):
            p = re.sub(r"\[.*\]", "", p)
            p = re.sub(r"\\n", " ", p)
            p = re.sub(r"\\t", " ", p)
            p = re.sub(r"\s+", " ", p)
            return p.strip()
        with open("data/wp/raw/train.wp_source", "r") as f:
            num_lines_read = 0
            print("Generating and writing WP prompts...")
            pbar = tqdm.tqdm(total=1000)
            for prompt in f:
                if num_lines_read >= 1000:
                    break
                input_prompt = format_prompt(prompt)
                inputs = neo_tokenizer(input_prompt, return_tensors="pt", max_length=50, truncation=True).to(device)
                outputs = neo_model.generate(**inputs, max_length=50, do_sample=True)
                reply = neo_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                with open(f"data/wp/prompts/{num_lines_read + 1}.txt", "w") as f:
                    f.write(reply)
                num_lines_read += 1
                pbar.update(1)
            pbar.close()

    if args.wp_human:
        print("Formatting Human WP documents...")
        with open("data/wp/raw/train.wp_target", "r") as f:
            num_lines_read = 0
            pbar = tqdm.tqdm(total=1000)
            for doc in f:
                if num_lines_read >= 1000:
                    break
                doc = doc.strip()
                tokens = doc.split(" ")
                replace = [["<newline>", "\n"]]
                for r in replace:
                    tokens = [t.replace(r[0], r[1]) for t in tokens]
                detokenizer = TreebankWordDetokenizer()
                formatted_doc = detokenizer.detokenize(tokens)
                formatted_doc = "\n".join([i.strip() for i in formatted_doc.split("\n")])
                formatted_doc = formatted_doc.replace("\n\n", "\n")
                formatted_doc = formatted_doc.replace(" .", ".")
                formatted_doc = formatted_doc.replace(" ’ ", "'")
                formatted_doc = formatted_doc.replace(" ”", '"')
                formatted_doc = formatted_doc.replace('"', '"')
                formatted_doc = formatted_doc.replace('"', '"')
                formatted_doc = html_replace(formatted_doc)
                with open(f"data/wp/human/{num_lines_read + 1}.txt", "w") as f:
                    f.write(formatted_doc)
                num_lines_read += 1
                pbar.update(1)
            pbar.close()

    if args.wp_gpt:
        print("Generating GPT WP documents...")
        for idx in tqdm.tqdm(range(1, 1001)):
            with open(f"data/wp/prompts/{idx}.txt", "r") as f:
                prompt = f.read().strip()
            with open(f"data/wp/human/{idx}.txt", "r") as f:
                words = round_to_100(len(f.read().split(" ")))
            prompts = get_wp_prompts(words, prompt)
            generate_documents(f"data/wp", prompts, prompt_types, neo_model, neo_tokenizer, device)

    if args.reuter_human:
        reuter_replace = ["--", "202-898-8312", "(", "($1=", "(A$", "Reuters Chicago"]
        authors = os.listdir("data/reuter/raw/C50train")
        print("Formatting Human Reuters documents...")
        for author in tqdm.tqdm(authors):
            if not os.path.exists(f"data/reuter/human/{author}"):
                os.makedirs(f"data/reuter/human/{author}")
            files = [
                f"data/reuter/raw/C50train/{author}/{i}"
                for i in os.listdir("data/reuter/raw/C50train/{author}")
            ] + [
                f"data/reuter/raw/C50test/{author}/{i}"
                for i in os.listdir("data/reuter/raw/C50test/{author}")
            ]
            for n, file in enumerate(files[:20]):
                with open(file, "r", encoding="utf-8") as f:
                    doc = f.read().strip()
                    doc = doc.replace("\n\n", "\n")
                    lines = doc.split("\n")
                    if any([i in lines[-1] for i in reuter_replace]):
                        lines = lines[:-1]
                    doc = "\n".join(lines)
                    doc = html_replace(doc)
                    with open(f"data/reuter/human/{author}/{n+1}.txt", "w") as f:
                        f.write(doc.strip())

    if args.reuter_gpt:
        print("Generating GPT Reuters documents...")
        authors = os.listdir("data/reuter/human")
        for author in tqdm.tqdm(authors):
            for idx in range(1, 21):
                with open(f"data/reuter/human/{author}/{idx}.txt", "r") as f:
                    words = round_to_100(len(f.read().split(" ")))
                with open(f"data/reuter/gpt/{author}/headlines/{idx}.txt", "r") as f:
                    headline = f.read().strip()
                prompts = get_reuter_prompts(words, headline)
                generate_documents(f"data/reuter/{author}", prompts, prompt_types, neo_model, neo_tokenizer, device)

    if args.essay_human or args.essay_gpt:
        essay_dataset = load_dataset("qwedsacf/ivypanda-essays")

    if args.essay_human:
        print("Formatting Human Essay documents...")
        num_documents, idx = 0, 0
        pbar = tqdm.tqdm(total=1000)
        while num_documents < 1000:
            essay = essay_dataset["train"][idx]
            essay = essay["TEXT"].strip()
            essay = essay[essay.index("\n") + 1:]
            idx += 1
            if "table of contents" in essay.lower():
                continue
            essay = essay.replace("\n\n", "\n")
            lines = essay.split("\n")
            doc = []
            for line in lines:
                if any(
                    [
                        i in line.lower()
                        for i in [
                            "references",
                            "reference",
                            "work cited",
                            "works cited",
                            "bibliography",
                        ]
                    ]
                ):
                    break
                doc.append(line)
            doc = "\n".join(doc)
            with open(f"data/essay/human/{num_documents + 1}.txt", "w") as f:
                f.write(doc.strip())
            num_documents += 1
            pbar.update(1)
        pbar.close()

    if args.essay_prompts:
        print("Generating Essay prompts...")
        for idx in tqdm.tqdm(range(1, 1001)):
            with open(f"data/essay/human/{idx}.txt", "r") as f:
                doc = f.read().strip()
            inputs = neo_tokenizer(" ".join(doc.split(" ")[:500]), return_tensors="pt", max_length=500, truncation=True).to(device)
            outputs = neo_model.generate(**inputs, max_length=50, do_sample=True)
            reply = neo_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            with open(f"data/essay/prompts/{idx}.txt", "w") as f:
                f.write(reply)

    if args.essay_gpt:
        print("Generating GPT Essay documents...")
        for type in prompt_types:
            if not os.path.exists(f"data/essay/{type}"):
                os.makedirs(f"data/essay/{type}")
        for idx in tqdm.tqdm(range(1, 1001)):
            with open(f"data/essay/prompts/{idx}.txt", "r") as f:
                prompt = f.read().strip()
            with open(f"data/essay/human/{idx}.txt", "r") as f:
                words = round_to_100(len(f.read().split(" ")))
            prompts = get_essay_prompts(words, prompt)
            generate_documents(f"data/essay", prompts, prompt_types, neo_model, neo_tokenizer, device)

    if args.logprobs:
        generate_logprobs(get_generate_dataset(*datasets))

    if args.logprob_other:
        other_datasets = [
            Dataset("normal", "data/other/ets"),
            Dataset("normal", "data/other/lang8"),
            Dataset("normal", "data/other/pelic"),
            Dataset("normal", "data/other/gptzero/gpt"),
            Dataset("normal", "data/other/gptzero/human"),
            Dataset("normal", "data/other/toefl91"),
            Dataset("normal", "data/other/undetectable"),
        ]
        generate_logprobs(get_generate_dataset(*other_datasets))

    if args.gen_perturb_char:
        def perturb_char_basic(doc, n=1):
            if len(doc) < 2:
                return doc
            for _ in range(n):
                perturb_type = np.random.choice(["swap", "delete", "insert"])
                if perturb_type == "swap":
                    idx = np.random.randint(len(doc) - 1)
                    doc = doc[:idx] + doc[idx + 1] + doc[idx] + doc[idx + 2:]
                elif perturb_type == "delete" and len(doc) > 1:
                    idx = np.random.randint(len(doc))
                    doc = doc[:idx] + doc[idx + 1:]
                elif perturb_type == "insert":
                    idx = np.random.randint(len(doc))
                    doc = (
                        doc[:idx]
                        + np.random.choice(list(string.ascii_letters))
                        + doc[idx:]
                    )
            return doc

        def perturb_char_space(doc, n=1):
            if len(doc) < 2:
                return doc
            for _ in range(n):
                perturb_type = np.random.choice(["insert", "delete"])
                if perturb_type == "insert":
                    idx = np.random.randint(len(doc))
                    doc = doc[:idx] + " " + doc[idx:]
                elif perturb_type == "delete":
                    space_indices = [
                        idx for idx, c in enumerate(doc) if c == " " or c == "\n"
                    ]
                    if len(space_indices) > 0:
                        idx = np.random.choice(space_indices)
                        doc = doc[:idx] + doc[idx + 1:]
            return doc

        def perturb_char_cap(doc, n=1):
            if len(doc) < 2:
                return doc
            for _ in range(n):
                idx = np.random.randint(len(doc))
                if doc[idx].isalpha():
                    if doc[idx].isupper():
                        doc = doc[:idx] + doc[idx].lower() + doc[idx + 1:]
                    else:
                        doc = doc[:idx] + doc[idx].upper() + doc[idx + 1:]
            return doc

        def perturb_word_adj(doc, n=1):
            words = doc.split(" ")
            if len(words) < 2:
                return doc
            for _ in range(n):
                idx = np.random.randint(len(words) - 1)
                words[idx], words[idx + 1] = words[idx + 1], words[idx]
            doc = " ".join(words)
            return doc

        def perturb_word_syn(doc, n=1):
            words = doc.split(" ")
            if len(words) < 2:
                return doc
            for _ in range(n):
                idx = np.random.randint(len(words))
                word = words[idx]
                synonym = closest_synonym(word)
                if synonym:
                    words[idx] = synonym
            doc = " ".join(words)
            return doc

        perturb_char_word_fns = {
            "char_basic": perturb_char_basic,
            "char_space": perturb_char_space,
            "char_cap": perturb_char_cap,
            "word_adj": perturb_word_adj,
            "word_syn": perturb_word_syn,
        }

        if not os.path.exists("data/perturb"):
            os.makedirs("data/perturb")

        np.random.seed(args.seed)
        indices = np.arange(6000)
        np.random.shuffle(indices)
        train, test = (
            indices[: math.floor(0.8 * len(indices))],
            indices[math.floor(0.8 * len(indices)):],
        )
        print("Train/Test Split:", train, test)
        files = generate_dataset_fn(lambda f: f, verbose=False)
        indices = np.arange(len(test))
        np.random.shuffle(indices)
        indices = indices[:200]
        labels = []
        for file in files[test][indices]:
            if "human" in file and "gpt" not in file:
                labels.append(0)
            elif "gpt" in file and "human" not in file:
                labels.append(1)
            else:
                raise ValueError("Invalid file name")
        with open("data/perturb/labels.txt", "w") as f:
            f.write("\n".join([str(i) for i in labels]))
        num_perturb = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200]
        for n in tqdm.tqdm(num_perturb):
            for perturb_type, func in perturb_char_word_fns.items():
                if not os.path.exists(f"data/perturb/{perturb_type}/{n}"):
                    os.makedirs(f"data/perturb/{perturb_type}/{n}")
                for idx, file in enumerate(files[test][indices]):
                    with open(file, "r", encoding="utf-8") as f:  # 加上 encoding
                        doc = f.read().strip()
                    perturb_doc = func(doc, n=n)
                    with open(f"data/perturb/{perturb_type}/{n}/{idx}.txt", "w", encoding="utf-8") as f:  # 写入也建议加 encoding
                        f.write(perturb_doc)

    if args.logprob_perturb_char:
        perturb_datasets = [
            Dataset("normal", f"data/perturb/{perturb_type}/{n}")
            for perturb_type in perturb_char_names
            for n in perturb_char_sizes
        ]
        generate_logprobs(get_generate_dataset(*perturb_datasets))

    if args.gen_perturb_sent:
        def perturb_sent_adj(doc, n=1):
            doc = nltk.sent_tokenize(doc)
            if len(doc) < 2:
                return (" ".join(doc)).strip()
            for _ in range(n):
                idx = np.random.randint(len(doc) - 1)
                doc[idx], doc[idx + 1] = doc[idx + 1], doc[idx]
            return (" ".join(doc)).strip()

        def perturb_sent_paraph(doc, n=1):
            doc = nltk.sent_tokenize(doc)
            if len(doc) < 1:
                return (" ".join(doc)).strip()
            for _ in range(n):
                idx = np.random.randint(len(doc))
                inputs = neo_tokenizer(doc[idx], return_tensors="pt", max_length=100, truncation=True).to(device)
                outputs = neo_model.generate(**inputs, max_length=100, do_sample=True)
                doc[idx] = neo_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return (" ".join(doc)).strip()

        def perturb_para_adj(doc, n=1):
            doc = doc.split("\n")
            if len(doc) < 2:
                return "\n".join(doc)
            for _ in range(n):
                idx = np.random.randint(len(doc) - 1)
                doc[idx], doc[idx + 1] = doc[idx + 1], doc[idx]
            return "\n".join(doc)

        def perturb_para_paraph(doc, n=1):
            doc = doc.split("\n")
            if len(doc) < 1:
                return "\n".join(doc)
            for _ in range(n):
                idx = np.random.randint(len(doc))
                inputs = neo_tokenizer(doc[idx], return_tensors="pt", max_length=100, truncation=True).to(device)
                outputs = neo_model.generate(**inputs, max_length=100, do_sample=True)
                doc[idx] = neo_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return "\n".join(doc)

        perturb_sent_fns = {
            "sent_adj": perturb_sent_adj,
            "sent_paraph": perturb_sent_paraph,
            "para_adj": perturb_para_adj,
            "para_paraph": perturb_para_paraph,
        }

        if not os.path.exists("data/perturb"):
            os.makedirs("data/perturb")

        np.random.seed(args.seed)
        indices = np.arange(6000)
        np.random.shuffle(indices)
        train, test = (
            indices[: math.floor(0.8 * len(indices))],
            indices[math.floor(0.8 * len(indices)):],
        )
        print("Train/Test Split:", train, test)
        files = generate_dataset_fn(lambda f: f, verbose=False)
        indices = np.arange(len(test))
        np.random.shuffle(indices)
        indices = indices[:200]
        labels = []
        for file in files[test][indices]:
            if "human" in file and "gpt" not in file:
                labels.append(0)
            elif "gpt" in file and "human" not in file:
                labels.append(1)
            else:
                raise ValueError("Invalid file name")
        with open("data/perturb/labels.txt", "w") as f:
            f.write("\n".join([str(i) for i in labels]))
        num_perturb = list(range(11))
        for n in tqdm.tqdm(num_perturb):
            for perturb_type, func in perturb_sent_fns.items():
                if not os.path.exists(f"data/perturb/{perturb_type}/{n}"):
                    os.makedirs(f"data/perturb/{perturb_type}/{n}")
                for idx, file in enumerate(files[test][indices]):
                    with open(file, "r") as f:
                        doc = f.read().strip()
                    perturb_doc = func(doc, n=n)
                    with open(f"data/perturb/{perturb_type}/{n}/{idx}.txt", "w") as f:
                        f.write(perturb_doc)

    if args.logprob_perturb_sent:
        perturb_datasets = [
            Dataset("normal", f"data/perturb/{perturb_type}/{n}")
            for perturb_type in perturb_sent_names
            for n in perturb_sent_sizes
        ]
        generate_logprobs(get_generate_dataset(*perturb_datasets))