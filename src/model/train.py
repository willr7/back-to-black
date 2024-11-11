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

import os
import torchmetrics

import warnings

from tqdm import tqdm
from pathlib import Path

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

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
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')

        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len); only hiding the padding tokens
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len); hidding the subsequent words

            # runs the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            # projection output, model output
            projection_output = model.project(decoder_output) # (B, seq_len, target_vocab_size) 

            label = batch['label'].to(device) # (B, seq_len) for each batch, we find the position in the vocabulary for a particular word label 

            # compare output to the label
            # turns (B, seq_len, target_vocab_size)  -> (B * seq_len, target_vocab_size) 
            loss = loss_fn(projection_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            # shows loss on progress bar
            batch_iterator.set_postfix({f'loss':f'{loss.item():6.3f}'})

            # record the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # backpropagate the loss
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step+=1
        # save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')   
        # good to store both model + optimizer since optimizer keeps track of stats 
        # for each weight and how to move each weight independently
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'global_step':global_step
        }, model_filename)


if __name__ == 'main':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)