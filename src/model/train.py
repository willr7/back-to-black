import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


from dataset.dataset import AAVE_SAE_Dataset, causal_mask
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_path".format(lang)])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    
    dataset_raw = load_dataset('Insert dataset name', f'{config['lang_src']}-{config['lang_tgt']}', split="train")

    # Build tokenizers
    tokenizer_source = get_or_build_tokenizer(config, dataset_raw, config['lang_src'])
    tokenizer_target = get_or_build_tokenizer(config, dataset_raw, config['lang_tgt']) 

    # Keep 90% for training and 10% for validation
    train_dataset_size = int(0.9 * len(dataset_raw))
    val_dataset_size = int(dataset_raw) - train_dataset_size
    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_dataset_size, val_dataset_size])

    train_dataset = AAVE_SAE_Dataset(train_dataset_raw, tokenizer_source, tokenizer_target, 
                                     config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_dataset = AAVE_SAE_Dataset(val_dataset_raw, tokenizer_source, tokenizer_target, 
                                   config['lang_src'], config['lang_tgt'], )