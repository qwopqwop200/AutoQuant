import random
import numpy as np
import torch
from datasets import load_dataset

def get_calib_dataset(tokenizer, name='wikitext2', nsamples=256, seqlen=512, seed=42):
    DATASET_MAP = {
        'wikitext2': get_wikitext2,
        'ptb': get_ptb,
        'ptb_new':get_ptb_new,
        'c4':get_c4,
        'pile':get_pile
    }
    
    if name not in DATASET_MAP:
        raise NotImplementedError(f"{name} isn't supported yet. Only support: {list(DATASET_MAP.keys())}")
    
    return DATASET_MAP[name](tokenizer, nsamples, seqlen, seed)

def get_wikitext2(tokenizer, nsamples=256, seqlen=512, seed=42):
    random.seed(seed)
    
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train') 
    enc = tokenizer("\n\n".join(data['text']), return_tensors='pt')

    calib_data = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        calib_data.append(inp)
    return calib_data

def get_ptb(tokenizer, nsamples=256, seqlen=512, seed=42):
    random.seed(seed)
    
    data = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    enc = tokenizer("\n\n".join(data['sentence']), return_tensors='pt')

    calib_data = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        calib_data.append(inp)
    return calib_data

def get_ptb_new(tokenizer, nsamples=256, seqlen=512, seed=42):
    random.seed(seed)
    
    data = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    enc = tokenizer(" ".join(data['sentence']), return_tensors='pt')

    calib_data = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        calib_data.append(inp)
    return calib_data

def get_c4(tokenizer, nsamples=256, seqlen=512, seed=42):
    random.seed(seed)
    
    data = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')

    calib_data = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(data) - 1)
            enc = tokenizer(data[i]['text'], return_tensors='pt')
            if enc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        calib_data.append(inp)
    return calib_data

def get_pile(tokenizer, nsamples=256, seqlen=512, seed=42):
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
        if total_len // seqlen >= nsamples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // seqlen
    calib_data = [cat_samples[:, i*seqlen:(i+1)*seqlen] for i in range(nsamples)]
    return calib_data