import random
import numpy as np
import torch
from datasets import load_dataset

def get_calib_dataset(tokenizer, name='wikitext2', n_samples=128, seq_length=512, seed=42):
    DATASET_MAP = {
        'wikitext2': get_wikitext2,
        'ptb': get_ptb,
        'ptb_new':get_ptb_new,
        'c4':get_c4,
        'pile':get_pile
    }
    
    if name not in DATASET_MAP:
        raise NotImplementedError(f"{name} isn't supported yet. Only support: {list(DATASET_MAP.keys())}")
    
    return DATASET_MAP[name](tokenizer, n_samples, seq_length, seed)

def get_wikitext2(tokenizer, n_samples=128, seq_length=512, seed=42):
    random.seed(seed)
    
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train') 
    enc = tokenizer("\n\n".join(data['text']), return_tensors='pt')

    calib_data = []
    for _ in range(n_samples):
        i = random.randint(0, enc.input_ids.shape[1] - seq_length - 1)
        j = i + seq_length
        inp = enc.input_ids[:, i:j]
        calib_data.append(inp)
    return calib_data

def get_ptb(tokenizer, n_samples=128, seq_length=512, seed=42):
    random.seed(seed)
    
    data = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    enc = tokenizer("\n\n".join(data['sentence']), return_tensors='pt')

    calib_data = []
    for _ in range(n_samples):
        i = random.randint(0, enc.input_ids.shape[1] - seq_length - 1)
        j = i + seq_length
        inp = enc.input_ids[:, i:j]
        calib_data.append(inp)
    return calib_data

def get_ptb_new(tokenizer, n_samples=128, seq_length=512, seed=42):
    random.seed(seed)
    
    data = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    enc = tokenizer(" ".join(data['sentence']), return_tensors='pt')

    calib_data = []
    for _ in range(n_samples):
        i = random.randint(0, enc.input_ids.shape[1] - seq_length - 1)
        j = i + seq_length
        inp = enc.input_ids[:, i:j]
        calib_data.append(inp)
    return calib_data

def get_c4(tokenizer, n_samples=128, seq_length=512, seed=42):
    random.seed(seed)
    
    data = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')

    calib_data = []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(data) - 1)
            enc = tokenizer(data[i]['text'], return_tensors='pt')
            if enc.input_ids.shape[1] >= seq_length:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seq_length - 1)
        j = i + seq_length
        inp = enc.input_ids[:, i:j]
        calib_data.append(inp)
    return calib_data

def get_pile(tokenizer, n_samples=128, seq_length=512, seed=42):
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=seed)
    samples = []
    total_len = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        total_len += sample.shape[1]
        samples.append(sample)
        if total_len // seq_length >= n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // seq_length
    calib_data = [cat_samples[:, i*seq_length:(i+1)*seq_length] for i in range(n_samples)]
    return calib_data