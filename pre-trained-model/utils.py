import csv
import random

import numpy as np


def print_random_decoded_entries(
    dataset,
    tokenizer,
    iteration,
    source_lang,
    target_lang,
    log_dir,
    log_predictions=True,
    num_rows=10,
):
    random_indices = random.sample(range(len(dataset)), num_rows)
    output_str = f"Iteration: {iteration}\n"

    for idx in random_indices:
        input_ids = dataset[idx]["input_ids"]
        labels = dataset[idx]["labels"]

        input_ids = [input_id for input_id in input_ids if input_id != -100]
        decoded_input_ids = tokenizer.decode(input_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.decode(labels, skip_special_tokens=True)

        output_str += "\n"
        output_str += f"Row {idx}:\n"
        output_str += f"  Predicted {source_lang}: {decoded_input_ids}\n"
        output_str += f"  Ground Truth {target_lang}: {decoded_labels}\n"

    output_str += "\n"

    print(output_str)

    if log_predictions:
        prediction_file_path = log_dir + "/predictions.txt"
        with open(prediction_file_path, "a") as f:
            f.writelines(output_str)


def fix_attention_mask(examples):
    examples["input_ids"] = [
        [x for x in input_ids if x != 0 and x != -100]
        for input_ids in examples["input_ids"]
    ]

    examples["attention_mask"] = [
        [1 for _ in input_ids] for input_ids in examples["input_ids"]
    ]

    return examples


def preprocess_source_to_target(
    examples, source_lang, target_lang, tokenizer, max_length=200
):
    inputs = examples[source_lang]
    targets = examples[target_lang]

    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )

    return model_inputs


def preprocess_source_to_target_function(
    examples, src_lang, tgt_lang, tokenizer, max_length=200
):
    inputs = examples[src_lang]
    targets = examples[tgt_lang]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs


def preprocess_target_to_source_function(
    examples, tgt_lang, src_lang, tokenizer, max_length=200
):
    inputs = examples[tgt_lang]
    targets = examples[src_lang]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs


def preprocess_lang_function(examples, src_lang, tokenizer, max_length=200):
    inputs = examples[src_lang]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
    return model_inputs


def yield_csv_lines(csv_dataset_path, source_lang, target_lang, n=1_000_000):
    with open(csv_dataset_path, "r") as csv_file:
        filereader = csv.reader(csv_file)
        for i, line in enumerate(filereader):
            if i >= n:
                break

            if line[0].strip() != "" and line[1].strip() != "":
                yield {source_lang: line[0], target_lang: line[1]}
            else:
                print("empty string found")


def yield_paired_lines(source_path, target_path, source_lang, target_lang):
    with (
        open(source_path, "r", encoding="utf-8") as source_text_file,
        open(target_path, "r", encoding="utf-8") as target_text_file,
    ):
        for source_line, target_line in zip(source_text_file, target_text_file):
            yield {source_lang: source_line, target_lang: target_line}


def yield_mono_lines(path, lang, n=1_000_000):
    with open(path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i >= n:
                break

            if line.strip() != "":
                yield {lang: line.strip()}
            else:
                print("empty string found")
