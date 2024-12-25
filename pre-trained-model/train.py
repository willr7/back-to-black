import evaluate
from datasets import Dataset, Sequence, Value, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import numpy as np
from utils import *
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Train translation models with iterative back-translation"
    )

    # Model paths
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="google-t5/t5-small",
        help="Name or path of the tokenizer to use",
    )
    parser.add_argument(
        "--source_to_target_model_name",
        type=str,
        default="google-t5/t5-small",
        help="Name or path of the base model to use",
    )
    parser.add_argument(
        "--target_to_source_model_name",
        type=str,
        default="google-t5/t5-small",
        help="Name or path of the base model to use",
    )

    # Data paths
    parser.add_argument(
        "--monolingual_src_path",
        type=str,
        default="/Users/willreed/projects/classes/nlp-final-project/cleaned_BAWE.txt",
        help="Path to source monolingual data",
    )
    parser.add_argument(
        "--monolingual_tgt_path",
        type=str,
        default="/Users/willreed/projects/classes/nlp-final-project/coraal_dataset.txt",
        help="Path to target monolingual data",
    )
    parser.add_argument(
        "--paired_csv_path",
        type=str,
        default="/Users/willreed/projects/classes/nlp-final-project/GPT-Translated-AAVE-Lyrics.csv",
        help="Path to paired CSV data",
    )

    # Experiment settings
    parser.add_argument(
        "--source_lang",
        type=str,
        default="SAE",
        help="Source language for translation",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default="AAVE",
        help="Target language for translation",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of back-translation iterations",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment (default: {iterations} iterations/)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for logs (default: ./logs/{experiment_name})",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of epochs to train in each iteration",
    )
    parser.add_argument(
        "--ratio", type=float, default=1, help="Ratio of synthetic data to real data"
    )

    args = parser.parse_args()

    # Set default experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.iterations} iterations"

        os.makedirs("logs", exist_ok=True)
        if args.experiment_name in os.listdir("./logs/"):
            n = 1
            args.experiment_name += f" {n}"
            while args.experiment_name in os.listdir("./logs/"):
                n += 1
                args.experiment_name = args.experiment_name[:-1] + str(n)
        print(args.experiment_name)

    # Set default log directory if not provided
    if args.log_dir is None:
        args.log_dir = f"./logs/{args.experiment_name}/"

    return args


args = get_args()

TOKENIZER = AutoTokenizer.from_pretrained(args.tokenizer_name)
SOURCE_TO_TARGET_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
    args.source_to_target_model_name
)
TARGET_TO_SOURCE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
    args.target_to_source_model_name
)

MONOLINGUAL_SRC_DATA_PATH = args.monolingual_src_path
MONOLINGUAL_TGT_DATA_PATH = args.monolingual_tgt_path
PAIRED_CSV_DATA_PATH = args.paired_csv_path

RATIO = args.ratio
NUM_EPOCHS = args.num_epochs
ITERATIONS = args.iterations

LOG_DIR = args.log_dir

SRC_LANG = args.source_lang
TGT_LANG = args.target_lang

METRIC = evaluate.load("sacrebleu")

DATA_COLLATOR = DataCollatorForSeq2Seq(TOKENIZER, model=SOURCE_TO_TARGET_MODEL)


def iterative_back_translation(
    parallel_data: Dataset,
    source_to_target_model: AutoModelForSeq2SeqLM,
    target_to_source_model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    source_data: Dataset,
    target_data: Dataset,
    iterations: int,
    source_lang: str,
    target_lang: str,
    num_epochs: int,
    log_dir: str,
    from_pretrained: bool = False,
    initial_training_epochs: int = 5,
):
    """
    Trains the source_to_target_model and the target_to_source_model using iterative back translation

    Parameter
    ========
    parallel_data: huggingface Dataset with two keys src_lang and tgt_lang

    source_to_target_model and target_to_source_model are huggingface Language Models for translation

    tokenizer is the tokenizer for the given language models

    source_data and target_data are huggingface datasets with one key, either src_lang or tgt_lang

    iterations is the number of iterations to do back translation

    source_lang and target_lang are the source and target languages

    num_epochs is the number of epochs to train both models in each step of back translation. Does not include initial model training

    from_pretrained indicates whether you are loading a pretrained dialect to dialect model (as opposed to the default t5 model)

    Returns
    =======
        the two models, trained
    """
    source_to_target_data = parallel_data
    target_to_source_data = parallel_data

    source_to_target_data = source_to_target_data.map(
        preprocess_source_to_target,
        batched=True,
        fn_kwargs={
            "source_lang": source_lang,
            "target_lang": target_lang,
            "tokenizer": tokenizer,
        },
    )
    target_to_source_data = source_to_target_data.map(
        preprocess_source_to_target,
        batched=True,
        fn_kwargs={
            "source_lang": target_lang,
            "target_lang": source_lang,
            "tokenizer": tokenizer,
        },
    )

    source_data = source_data.map(
        preprocess_lang_function,
        batched=True,
        fn_kwargs={"src_lang": source_lang, "tokenizer": tokenizer},
    )
    target_data = target_data.map(
        preprocess_lang_function,
        batched=True,
        fn_kwargs={"src_lang": target_lang, "tokenizer": tokenizer},
    )

    if not from_pretrained:
        target_to_source_model, target_to_source_trainer = train_model(
            model=target_to_source_model,
            data=target_to_source_data,
            source_lang=target_lang,
            target_lang=source_lang,
            num_epochs=initial_training_epochs,
            root_log_dir=log_dir,
            iteration=0,
        )

    for iteration in range(1, iterations + 1):
        print(f"Starting iteration: {iteration}")
        print(
            f"Generating synthetic {source_lang} data from monolingual {target_lang} data"
        )

        # Generate synthetic source data
        synthetic_source_data = target_to_source_trainer.predict(
            test_dataset=target_data, max_length=40
        ).predictions.tolist()

        # combine datasets
        synthetic_source_to_target_data = target_data.rename_column(
            "input_ids", "labels"
        )
        synthetic_source_to_target_data = synthetic_source_to_target_data.add_column(
            "input_ids",
            synthetic_source_data,
        )

        synthetic_source_to_target_data = synthetic_source_to_target_data.cast_column(
            "labels", Sequence(Value("int64"))
        )
        synthetic_source_to_target_data = synthetic_source_to_target_data.cast_column(
            "input_ids", Sequence(Value("int32"))
        )

        combined_source_to_target_data = concatenate_datasets(
            [source_to_target_data, synthetic_source_to_target_data]
        )

        combined_source_to_target_data = combined_source_to_target_data.map(
            fix_attention_mask, batched=True
        )

        source_to_target_model, source_to_target_trainer = train_model(
            model=source_to_target_model,
            data=combined_source_to_target_data,
            source_lang=source_lang,
            target_lang=target_lang,
            num_epochs=num_epochs,
            root_log_dir=log_dir,
            iteration=iteration,
        )

        print(f"Iteration: {iteration}")
        print(
            f"Generating synthetic {target_lang} data from monolingual {source_lang} data"
        )

        # generate synthetic target data and combine datasets
        synthetic_target_data = source_to_target_trainer.predict(
            test_dataset=source_data, max_length=40
        ).predictions.tolist()

        synthetic_target_to_source_data = source_data.rename_column(
            "input_ids", "labels"
        )
        synthetic_target_to_source_data = synthetic_target_to_source_data.add_column(
            "input_ids", synthetic_target_data
        )

        synthetic_target_to_source_data = synthetic_target_to_source_data.cast_column(
            "labels", Sequence(Value("int64"))
        )
        synthetic_target_to_source_data = synthetic_target_to_source_data.cast_column(
            "input_ids", Sequence(Value("int32"))
        )

        combined_target_to_source_data = concatenate_datasets(
            [target_to_source_data, synthetic_target_to_source_data]
        )

        combined_target_to_source_data = combined_target_to_source_data.map(
            fix_attention_mask, batched=True
        )

        target_to_source_model, target_to_source_trainer = train_model(
            model=target_to_source_model,
            data=combined_target_to_source_data,
            source_lang=target_lang,
            target_lang=source_lang,
            num_epochs=num_epochs,
            root_log_dir=log_dir,
            iteration=iteration,
        )

    return source_to_target_model, target_to_source_model


def train_model(
    model: AutoModelForSeq2SeqLM,
    data: Dataset,
    source_lang: str,
    target_lang: str,
    num_epochs: int,
    root_log_dir: str,
    iteration: int,
) -> tuple[AutoModelForSeq2SeqLM, Seq2SeqTrainer]:
    log_dir = root_log_dir + f"{source_lang}_to_{target_lang}/iteration {iteration}"
    output_dir = (
        root_log_dir + f"{source_lang}_to_{target_lang}_models/iteration {iteration}"
    )

    data = data.train_test_split(test_size=0.1)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=log_dir,
        logging_steps=500,
        predict_with_generate=True,
        # fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        data_collator=DATA_COLLATOR,
        tokenizer=TOKENIZER,
        compute_metrics=compute_metrics,
    )

    print(f"Iteration: {iteration}")
    print(f"Training {source_lang} to {target_lang} model")

    trainer.train()

    save_test_predictions(
        trainer=trainer,
        dataset=data["test"],
        tokenizer=TOKENIZER,
        source_lang=source_lang,
        target_lang=target_lang,
        iteration=iteration,
        log_dir=log_dir,
        top_k=10,
        bottom_k=10,
    )

    return model, trainer


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = TOKENIZER.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, TOKENIZER.pad_token_id)
    decoded_labels = TOKENIZER.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = METRIC.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != TOKENIZER.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    return result


def main():
    raw_paired_dataset = Dataset.from_generator(
        yield_csv_lines,
        gen_kwargs={
            "csv_dataset_path": PAIRED_CSV_DATA_PATH,
            "source_lang": SRC_LANG,
            "target_lang": TGT_LANG,
            # use n for debugging
            # only loads n samples
            "n": 1000,
        },
    )

    size_paired_dataset = len(raw_paired_dataset)

    raw_monolingual_src_data = Dataset.from_generator(
        yield_mono_lines,
        gen_kwargs={
            "path": MONOLINGUAL_SRC_DATA_PATH,
            "lang": SRC_LANG,
            # "n": RATIO * size_paired_dataset,
        },
    )
    raw_monolingual_tgt_data = Dataset.from_generator(
        yield_mono_lines,
        gen_kwargs={
            "path": MONOLINGUAL_TGT_DATA_PATH,
            "lang": TGT_LANG,
            # "n": RATIO * size_paired_dataset,
        },
    )

    raw_monolingual_src_data = raw_monolingual_src_data.shuffle().select(
        range(max(len(raw_monolingual_src_data), RATIO * size_paired_dataset))
    )
    raw_monolingual_tgt_data = raw_monolingual_tgt_data.shuffle().select(
        range(max(len(raw_monolingual_tgt_data), RATIO * size_paired_dataset))
    )

    iterative_back_translation(
        parallel_data=raw_paired_dataset,
        source_to_target_model=SOURCE_TO_TARGET_MODEL,
        target_to_source_model=TARGET_TO_SOURCE_MODEL,
        tokenizer=TOKENIZER,
        source_data=raw_monolingual_src_data,
        target_data=raw_monolingual_tgt_data,
        iterations=ITERATIONS,
        source_lang=SRC_LANG,
        target_lang=TGT_LANG,
        num_epochs=NUM_EPOCHS,
        log_dir=LOG_DIR,
    )


if __name__ == "__main__":
    main()
