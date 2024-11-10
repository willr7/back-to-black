import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


from dataset.dataset import AAVE_SAE_Dataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter


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
                                   config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    max_len_src = 0
    max_len_target = 0

    for item in dataset_raw:
        source_ids = tokenizer_source.encode(item['translation'][ config['lang_src'] ]).ids
        target_ids = tokenizer_source.encoder(item['translation'][ config['lang_tgt'] ]).ids
        max_len_src = max(max_len_src, len(source_ids))
        max_len_target = max(max_len_target, len(target_ids))

    print(f'Max length of source ids: {max_len_src}')
    print(f'Max length of target ids: {max_len_target}')

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_source, tokenizer_target

def get_model(config, vocab_source_len, vocab_target_len):

    # builds a transformer model
    model = build_transformer(vocab_source_len, vocab_target_len, 
                              config['seq_len'], config['seq_len'], config['d_model'])
    
    return model

def train_model(config):
    # define the device
    device = torch.device("cuda" if torch.cuda_is_available() else 'cpu')
    print(f"Using device {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # restore the state of the model/optimizer if the model crashes
    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model: {model_filename}')
        state = torch.load(model_filename)

        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropy(ignore_index=tokenizer_src.token_to_id(['PAD']), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        
        # 
        batch_iterator = tqdm(train_dataloader)