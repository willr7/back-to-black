import torch
from torch.utils.data import Dataset
import typing
import os
import pandas as pd
import re

class Source_Target_Dataset(Dataset):

    def __init__(
        self,
        dataset,
        tokenizer_src,
        tokenizer_target,
        source_lang,
        target_lang,
        seq_len,
    ) -> None:
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

        self.sos_token = torch.tensor(
            [tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: typing.Any) -> typing.Any:

        src_target_pair = self.dataset[index]
        source_text = src_target_pair[self.source_lang]
        target_text = src_target_pair[self.target_lang]

        # array of input ids, #'s that correspond to each word
        encoder_input_tokens = self.tokenizer_src.encode(source_text).ids
        # array of output ids
        decoder_input_tokens = self.tokenizer_target.encode(target_text).ids

        encoder_num_pad_tokens = (
            self.seq_len - len(encoder_input_tokens) - 2
        )  # including EOS and SOS token
        decoder_num_pad_tokens = (
            self.seq_len - len(decoder_input_tokens) - 1
        )  # only adds either EOS or SOS token

        if encoder_num_pad_tokens < 0 or decoder_num_pad_tokens < 0:
            raise ValueError(
                "Negative number of encoder or decoder padding tokens, so sentence is too long!!"
            )

        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * encoder_num_pad_tokens, dtype=torch.int64
                ),
            ]
        )

        # Add SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * decoder_num_pad_tokens, dtype=torch.int64
                ),
            ]
        )

        # Add EOS to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * decoder_num_pad_tokens, dtype=torch.int64
                ),
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            # we do not want padding tokens in self-attention mechanism so we use masking for encoder and decoder
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1, 1, seq_len)
            # causal mask
            "decoder_mask": (decoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & causal_mask(
                decoder_input.size(0)
            ),  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,  # (seq_len)
            "src_text": source_text,
            "tgt_text": target_text,
        }


def causal_mask(size):
    # gets the upper trianglar matrix above diagonal
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
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

def create_coraal_content_dataframe(column_name, column_idx, num_columns, data):
    """
    Columns in the coraal dataset => 'Line', 'Spkr', 'StTime', 'Content', 'EnTime'
    The index for the different columns in the coraal dataset 
    """
    # only gets the data from ta specify column
    data_list = []
    characters_to_replace = [",", ";", "(",")", ":", ".", "!", "pause", "laugh", "[", 
                             "]", "RD-NAME", "1", "2", "3", "4", "5", "6", "7", 
                             "8", "9","0", "-", "/"]

    for i in range(num_columns, len(data)-1, num_columns):
        # print("i: ", i)
        text = data[i+column_idx]
        print("before splitting: ", text)

        # removes uncessary characters
        for char in characters_to_replace:
            text = text.replace(char, "")

        print("after splitting: ", text)
        # skips any text that only has replaceable characters
        if len(text) == 0 or text.strip() == "":
            continue
        
        data_list.append(text)

    # return pd.DataFrame({column_name: [ ''.join(re.split(",|", data[i+column_idx])) for i in range(num_columns, len(data)-1, num_columns)]})
    return pd.DataFrame({column_name: data_list})

CONTENT_COLUMN_IDX = 3
COLUMN_NAME = "Content"
def coral_preprocessing(file_directory, data_column_name="Content"):
    """
    data_column_name: str
        name of the column with the transcripted words
    """

    filename_list = []

    for filename in os.listdir(file_directory):
        file_path = file_directory + "/" + filename
        # print("filename: ", filename)
        # print("file_path: ", file_path)

        filename_list.append(file_path)

    text_file_list = []
    # print("getting the text file path")
    for filename in filename_list:
        for text_file in os.listdir(filename):
            text_file_name = filename + "/" + text_file
            text_file_list.append(text_file_name)
            # print("text file path: ", text_file_name)
    
    content_list = []

    for text_file_path in text_file_list:
        
        try:
            with open(text_file_path, 'r') as file:
                content = file.read()
                content_list.append(content)
        except:
            with open(text_file_path, 'rb') as binary_file:
                binary_content = binary_file.read()
                content_list.append(binary_content)

    elements = content_list[:20]
    list_content_df = []
    for idx, c in enumerate(elements):
        split_content = re.split("\t|\n", c)
        # print("content")
        # print(c.split("\t"))
        # print(split_content)
        # there are 5 columns
        # print("length of all the content in the coraal textfile: ", len(split_content))
        # print("is the data a multiple of 5: ", len(split_content)%5 == 0)
        
        content_df = create_coraal_content_dataframe(data_column_name, CONTENT_COLUMN_IDX, 5, split_content)
        list_content_df.append(content_df)

    return pd.concat(list_content_df)
        
# src\model\coraal_data\ATL_textfiles_2020.05\ATL_se0_ag1_f_01_1.txt
# directory_name = "/coraal_data"
# coral_preprocessing(directory_name)

