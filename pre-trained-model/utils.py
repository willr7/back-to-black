import csv
import os

from typing import List, Dict, Any

import evaluate
from transformers import Seq2SeqTrainer, AutoTokenizer
from datasets import Dataset


def compute_bleu_scores(
    predictions: List, references: List, metric: evaluate.EvaluationModule
) -> List:
    """
    Computes a sentence-level BLEU score using sacrebleu.
    Returns a float (the BLEU score).
    """
    references = [[reference] for reference in references]
    # sacrebleu expects list of predictions and list of lists of references
    scores = []
    for prediction, reference in zip(predictions, references):
        result = metric.compute(predictions=[prediction], references=[[reference]])
        scores.append(result["score"])
    return scores


def save_test_predictions(
    trainer: Seq2SeqTrainer,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    source_lang: str,
    target_lang: str,
    iteration: int,
    log_dir: str,
    top_k: int = 10,
    bottom_k: int = 10,
) -> None:
    """
    :param trainer: Your Seq2SeqTrainer
    :param dataset: The dataset split you want to predict on (e.g., data["test"])
    :param tokenizer: The tokenizer to decode input_ids/labels/predictions
    :param source_lang: A string for the source language name (for logging)
    :param target_lang: A string for the target language name (for logging)
    :param iteration: The iteration number (for logging)
    :param log_dir: Directory where the output files will be saved
    :param top_k: How many of the highest BLEU samples to save
    :param bottom_k: How many of the lowest BLEU samples to save
    """
    label_ids = dataset["labels"]
    input_ids = dataset["input_ids"]

    predictions_output = trainer.predict(
        test_dataset=dataset.remove_columns("labels"), max_length=40
    )
    predicted_ids = predictions_output.predictions  # shape: [batch_size, seq_len]

    predicted_ids = [
        [predicted_id for predicted_id in predicted_id_sequence if predicted_id != -100]
        for predicted_id_sequence in predicted_ids
    ]

    decoded_predictions = tokenizer.batch_decode(
        predicted_ids, skip_special_tokens=True
    )

    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    decoded_input_ids = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    metric = evaluate.load("sacrebleu")
    bleu_scores = compute_bleu_scores(decoded_predictions, decoded_labels, metric)

    results: List[Dict[str, Any]] = []

    for idx in range(len(dataset)):
        source_text = decoded_input_ids[idx]
        label = decoded_labels[idx]
        predicted_text = decoded_predictions[idx]
        bleu_score = bleu_scores[idx]
        results.append(
            {
                "source": source_text,
                "label": label,
                "prediction": predicted_text,
                "bleu": bleu_score,
            }
        )

    all_preds_file_path = os.path.join(log_dir, "all_test_predictions.txt")
    with open(all_preds_file_path, "w", encoding="utf-8") as f:
        avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
        header = f"Iteration: {iteration} - All Test Predictions\n\nBleu Score: {avg_bleu_score:.2f}\n\n"
        f.write(header)
        for i, entry in enumerate(results):
            f.write(f"Row {i}:\n")
            f.write(f"  Source ({source_lang}):     {entry['source']}\n")
            f.write(f"  Label ({target_lang}):     {entry['label']}\n")
            f.write(f"  Prediction ({source_lang}): {entry['prediction']}\n")
            f.write(f"  BLEU: {entry['bleu']:.2f}\n\n")

    sorted_by_bleu = sorted(results, key=lambda x: x["bleu"], reverse=True)

    highest_bleu_samples_path = os.path.join(log_dir, "highest_bleu_samples.txt")
    with open(highest_bleu_samples_path, "w", encoding="utf-8") as f:
        header = f"Iteration: {iteration} - Top {top_k} Highest BLEU Samples\n\n"
        f.write(header)
        for i, entry in enumerate(sorted_by_bleu[:top_k]):
            f.write(f"Rank {i+1} (BLEU: {entry['bleu']:.2f})\n")
            f.write(f"  Source ({source_lang}):     {entry['source']}\n")
            f.write(f"  Label ({target_lang}):     {entry['label']}\n")
            f.write(f"  Prediction ({source_lang}): {entry['prediction']}\n\n")

    lowest_bleu_samples_path = os.path.join(log_dir, "lowest_bleu_samples.txt")
    with open(lowest_bleu_samples_path, "w", encoding="utf-8") as f:
        header = f"Iteration: {iteration} - Bottom {bottom_k} Lowest BLEU Samples\n\n"
        f.write(header)
        for i, entry in enumerate(sorted_by_bleu[-bottom_k:]):
            f.write(
                f"Rank {len(sorted_by_bleu) - bottom_k + i + 1} (BLEU: {entry['bleu']:.2f})\n"
            )
            f.write(f"  Source ({source_lang}):     {entry['source']}\n")
            f.write(f"  Label ({target_lang}):     {entry['label']}\n")
            f.write(f"  Prediction ({source_lang}): {entry['prediction']}\n\n")

    print(
        f"\nAll test predictions saved to: {all_preds_file_path}\n"
        f"Highest BLEU samples saved to: {highest_bleu_samples_path}\n"
        f"Lowest BLEU samples saved to:  {lowest_bleu_samples_path}\n"
    )


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
