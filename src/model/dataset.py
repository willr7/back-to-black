import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import sys

class AAVE_SAE_Dataset(Dataset):

    def __init__(self, dataset, tokenizer_src, tokenizer_target, source_lang, target_lang, seq_len) -> None:
        """
        Constructor to initialize AAVE_SAE_Dataset with the dataset, tokenizer for source and target language,
        source language, target language, sos token, eos token, and padding token

        Return 
        """
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_target = tokenizer_target
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([ tokenizer_src.token_to_id('[SOS]') ], dtype=torch.int64)
        self.eos_token = torch.tensor([ tokenizer_src.token_to_id('[EOS]') ], dtype=torch.int64)
        self.pad_token = torch.tensor([ tokenizer_src.token_to_id('[PAD]') ], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index:any) -> any:

        src_target_pair = self.dataset[index]
        source_text = src_target_pair[self.source_lang]
        target_text = src_target_pair[self.target_lang]

        # array of input ids, #'s that correspond to each word
        encoder_input_tokens = self.tokenizer_src.encode(source_text).ids
        # array of output ids
        decoder_input_tokens = self.tokenizer_target.encode(target_text).ids
        
        encoder_num_pad_tokens = self.seq_len - len(encoder_input_tokens) - 2 # including EOS and SOS token
        decoder_num_pad_tokens = self.seq_len - len(decoder_input_tokens) - 1 # only adds either EOS or SOS token

        if encoder_num_pad_tokens < 0 or decoder_num_pad_tokens < 0:
            raise ValueError("Negative number of encoder or decoder padding tokens, so sentence is too long!!")
        
        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encoder_num_pad_tokens, dtype=torch.int64)
            ]
        )

        # Add SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * decoder_num_pad_tokens, dtype=torch.int64)
            ]
        )

        # Add EOS to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decoder_num_pad_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input" : encoder_input, # (seq_len)
            "decoder_input" : decoder_input, # (seq_len) 
            # we do not want padding tokens in self-attention mechanism so we use masking for encoder and decoder
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            # causal mask
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label" : label, # (seq_len)
            "src_text" : source_text,
            "tgt_text" : target_text
        }

def causal_mask(size):
    # gets the upper trianglar matrix above diagonal
    mask = torch.triu( torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def coral_preprocessing(file_directory, data_column_name="Content"):
    """
    data_column_name: str
        name of the column with the transcripted words
    """

    filename_list = []

    for filename in os.listdir(file_directory):
        file_path = file_directory + "/" + filename
        print("filename: ", filename)
        print("file_path: ", file_path)

        filename_list.append(file_path)
    
    content_list = []

    for filename in filename_list:
        
        try:
            with open(filename, 'r') as file:
                content = file.read()
                content_list.append(content)
        except:
            with open(filename, 'rb') as binary_file:
                binary_content = binary_file.read()
                content_list.append(binary_content)

        
# src\model\coraal_data\ATL_textfiles_2020.05\ATL_se0_ag1_f_01_1.txt
directory_name = "/coraal_data"
coral_preprocessing(directory_name)

