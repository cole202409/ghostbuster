import os
import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_documents(output_dir, prompts, prompt_types, model, tokenizer, device, verbose=True, force_regenerate=False):
    if not os.path.exists(f"{output_dir}/logprobs"):
        os.mkdir(f"{output_dir}/logprobs")
    if verbose:
        print("Generating Articles...")
    for idx, (prompt, prompt_type) in enumerate(zip(tqdm.tqdm(prompts) if verbose else prompts, prompt_types)):
        output_path = f"{output_dir}/{prompt_type}/{idx}.txt"
        if os.path.exists(output_path) and not force_regenerate:
            continue
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        outputs = model.generate(**inputs, max_length=512, do_sample=True, num_return_sequences=1)
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        reply = reply.replace("\n\n", "\n")
        with open(output_path, "w") as f:
            f.write(reply)
    if verbose:
        print("Writing logprobs...")
    for idx, prompt_type in enumerate(tqdm.tqdm(prompt_types) if verbose else prompt_types):
        doc_path = f"{output_dir}/{prompt_type}/{idx}.txt"
        with open(doc_path) as f:
            doc = f.read().strip()
        neo_logprob_path = f"{output_dir}/logprobs/{idx}-neo.txt"
        gpt2_logprob_path = f"{output_dir}/logprobs/{idx}-gpt2.txt"
        if not os.path.exists(neo_logprob_path) and not force_regenerate:
            from utils.write_logprobs import write_logprobs
            write_logprobs(doc, neo_logprob_path, model, tokenizer)
        if not os.path.exists(gpt2_logprob_path) and not force_regenerate:
            gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
            write_logprobs(doc, gpt2_logprob_path, gpt2_model, gpt2_tokenizer)

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

            # 只用 write_logprobs 写入
            neo_file = convert_file_to_logprob_file(file, "neo")
            if not os.path.exists(neo_file):
                write_logprobs(doc, neo_file, neo_model, neo_tokenizer)
            gpt2_file = convert_file_to_logprob_file(file, "gpt2")
            if not os.path.exists(gpt2_file):
                write_logprobs(doc, gpt2_file, gpt2_model, gpt2_tokenizer)

        except Exception as e:
            print(f"[Warning] Skipping file {file} due to error: {e}")