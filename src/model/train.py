import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import Dataset as HuggingFaceDataset

from dataset import Source_Target_Dataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

import os
import torchmetrics

import warnings
from tqdm import tqdm
from pathlib import Path

SOURCE_LANGUAGE = "AAVE"
TARGET_LANGUAGE = "SAE"

def beam_search(model, beam_size, encoder_input, encoder_mask, tokenizer_source, tokenizer_target, max_len, device):
    """
    Performs beam search on the encoder
    
    Parameters:
    encoder_input: Torch.Tensor
        encoded input source sentence
    
    encoder_mask: Torch.Tensor
        binary vector indicating which elements of the encoder's output should be considered during decoding process

    Return:
    """
    sos_idx = tokenizer_target.token_to_id(['SOS'])
    eos_idx = tokenizer_target.token_to_id(['EOS'])

    # precompute the encoder output
    encoder_output = model.encode(encoder_input, encoder_mask)
    # initialize decoder input with the sos token with the same type as the encoder input
    decoder_initial_input = torch.empty(1,1).fill_(sos_idx).type_as(encoder_input).to(device)

    # create a beam list
    beam_list = [(decoder_initial_input, 1)]

    while True:

        # checking for any beam that has reached max length
            # this means we have run the decoding for at least max_len iterations, so stop the beam search
        if any([beam.size(1) == max_len for beam, _ in beam_list]):
            break

        new_beams = []

        # explores k * k possible beams and then only move forward with top k beams
        for beam, score in beam_list:

            # checking if the eos token has been reached
            if beam[0][-1] == eos_idx:
                continue

            # build beam's mask 
            beam_mask = causal_mask(beam.size(1)).type_as(encoder_mask).to(device)

            # calculate output
            output = model.decode(encoder_output, encoder_mask, beam, beam_mask)
            # get next token probabilities (score)
            prob = model.project(output[:, -1])

            # get top k beams
            top_k_prob, top_k_idx = torch.topk(prob, beam_size, dim=1)

            for i in range(beam_size):

                # for each of the top k beams, get the token and its probability
                token = top_k_idx[0][i].unsqueeze(0).unsqueeze(0)    
                token_prob = top_k_prob[0][i].item()

                # create new beam by appending token to current beam
                new_beam = torch.cat([beam, token], dim=1)
                # sum the log probabilities cuz' probabilities in log space 
                # (adding in log space => multiplying in normal base) 
                new_beams.append((new_beam, score + token_prob))

        # sort the new beams by their score value
        beam_list = sorted(new_beams, key=lambda x: x[1], reverse=True)
        # keeps the top k beams
        beam_list = beam_list[:beam_size]

        # If all the beams have reached the eos token, stop
        if all([beam[0][-1].item() == eos_idx for beam, _ in beam_list]):
            break

    # return the best beam after beam search
    return beam_list[0][0].squeeze()

def greedy_decode(model, sentence, tokenizer_src, tokenizer_tgt, max_len, device):
    with torch.no_grad():
        source = tokenizer_src.encode(sentence)
        source = torch.cat(
            [
                torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64),
                torch.tensor(source.ids, dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64),
                torch.tensor(
                    [tokenizer_src.token_to_id("[PAD]")]
                    * (max_len - len(source.ids) - 2),
                    dtype=torch.int64,
                ),
            ],
            dim=0,
        ).to(device)

        source_mask = (
            (source != tokenizer_src.token_to_id("[PAD]"))
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            .to(device)
        )

        encoder_output = model.encode(source, source_mask)
        decoder_input = (
            torch.empty(1, 1)
            .fill_(tokenizer_tgt.token_to_id("[SOS]"))
            .type_as(source)
            .to(device)
        )

        while True:
            if decoder_input.size(1) == max_len:
                break

            decoder_mask = (
                causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            )

            decoder_output = model.decode(
                encoder_output, source_mask, decoder_input, decoder_mask
            )

            log_probs = model.project(decoder_output[:, -1])

            top_predictions = torch.topk(log_probs, k=3)

            next_word = top_predictions.indices[0, 0]

            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.empty(1, 1)
                    .type_as(source)
                    .fill_(next_word.item())
                    .to(device),
                ],
                dim=1,
            )

            if next_word.item() == 3:
                break

    return decoder_input.squeeze(0)

def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_step,
    writer,
    num_examples=2,
):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model,
                batch["src_text"][0],
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg("-" * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg("-" * console_width)
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar("validation cer", cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar("validation wer", wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar("validation BLEU", bleu, global_step)
        writer.flush()

def get_all_sentences(ds, lang):
    for item in ds[lang]:
        yield item

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            vocab_size=500,
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def load_source_target_dataset(source_path, target_path, source_lang, target_lang):
    with open(source_path, "r", encoding="utf-8") as f:
        source_texts = f.readlines()
    with open(target_path, "r", encoding="utf-8") as f:
        target_texts = f.readlines()

    # Check lengths match
    assert len(source_texts) == len(target_texts), "Mismatch in line counts between files"

    # Create dataset dictionary
    return {source_lang: source_texts, target_lang: target_texts}

def get_ds(config):
    # dataset_raw = load_dataset('Insert dataset name', f'{config["lang_src"]}-{config["lang_tgt"]}', split="train")

    source_file_path = f"{config['data_folder']}{config['lang_src']}_samples.txt"
    target_file_path = f"{config['data_folder']}{config['lang_tgt']}_samples.txt"

    dataset_raw = load_source_target_dataset(source_file_path, target_file_path, SOURCE_LANGUAGE, TARGET_LANGUAGE)

    # Build tokenizers
    tokenizer_source = get_or_build_tokenizer(config, dataset_raw, config["lang_src"])
    tokenizer_target = get_or_build_tokenizer(config, dataset_raw, config["lang_tgt"])

    # Keep 90% for training and 10% for validation
    train_dataset_size = int(0.9 * len(dataset_raw))
    val_dataset_size = int(len(dataset_raw)) - train_dataset_size
    train_dataset_raw, val_dataset_raw = random_split(
        dataset_raw, [train_dataset_size, val_dataset_size]
    )

    train_dataset = Source_Target_Dataset(
        train_dataset_raw,
        tokenizer_source,
        tokenizer_target,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_dataset = Source_Target_Dataset(
        val_dataset_raw,
        tokenizer_source,
        tokenizer_target,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    max_len_src = 0
    max_len_target = 0
    for item in dataset_raw:
        source_ids = tokenizer_source.encode(item[config["lang_src"]]).ids
        target_ids = tokenizer_source.encode(item[config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(source_ids))
        max_len_target = max(max_len_target, len(target_ids))

    print(f"Max length of source ids: {max_len_src}")
    print(f"Max length of target ids: {max_len_target}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_source, tokenizer_target


def get_model(config, vocab_source_len, vocab_target_len):
    # builds a transformer model
    model = build_transformer(
        vocab_source_len,
        vocab_target_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )
    return model


def train_model(config):
    # define the device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Using device:", device)
    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(device)}")
        print(
            f"Device memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3} GB"
        )
    elif device == "mps":
        print("Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print(
            "      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc"
        )
        print(
            "      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu"
        )
    device = torch.device(device)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    # restore the state of the model/optimizer if the model crashes
    initial_epoch = 0
    global_step = 0

    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model: {model_filename}")
        state = torch.load(model_filename)

        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (B, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (B, seq_len)
            encoder_mask = batch["encoder_mask"].to(
                device
            )  # (B, 1, 1, seq_len); only hiding the padding tokens
            decoder_mask = batch["decoder_mask"].to(
                device
            )  # (B, 1, seq_len, seq_len); hidding the subsequent words

            # runs the tensors through the transformer
            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (B, seq_len, d_model)

            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (B, seq_len, d_model)

            # projection output, model output
            projection_output = model.project(
                decoder_output
            )  # (B, seq_len, target_vocab_size)

            label = batch["label"].to(
                device
            )  # (B, seq_len) for each batch, we find the position in the vocabulary for a particular word label

            # compare output to the label
            # turns (B, seq_len, target_vocab_size)  -> (B * seq_len, target_vocab_size)
            loss = loss_fn(
                projection_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1),
            )
            # shows loss on progress bar
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # record the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # backpropagate the loss
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        # good to store both model + optimizer since optimizer keeps track of stats
        # for each weight and how to move each weight independently
        run_validation(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config["seq_len"],
            device,
            print,
            global_step,
            writer,
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


def translate(sentence: str):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(device)}")
        print(
            f"Device memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3} GB"
        )
    elif device == "mps":
        print(f"Device name: <mps>")

    device = torch.device(device)

    config = get_config()

    tokenizer_src = Tokenizer.from_file(
        str(Path(config["tokenizer_file"].format(config["lang_src"])))
    )
    tokenizer_tgt = Tokenizer.from_file(
        str(Path(config["tokenizer_file"].format(config["lang_tgt"])))
    )

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    ).to(device)

    model_filename = get_weights_file_path(config, config["preload"])
    print(f"Loading model: {model_filename}")
    state = torch.load(model_filename)

    model.load_state_dict(state["model_state_dict"])

    seq_len = config["seq_len"]

    model.eval()
    model_out = greedy_decode(
        model, sentence, tokenizer_src, tokenizer_tgt, seq_len, device
    )
    model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

    print(model_out_text)


if __name__ == "__main__":
    # warnings.filterwarnings('ignore')
    # aave_file_path = "../data/aave_samples.txt"
    # sae_file_path = "../data/sae_samples.txt"
    config = get_config()
    train_model(config)
    # translate("tryna get this out my headbut I can't help it")
