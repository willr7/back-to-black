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


TOKENIZER = AutoTokenizer.from_pretrained("google-t5/t5-small")
SOURCE_TO_TARGET_MODEL = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
TARGET_TO_SOURCE_MODEL = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

EXPERIMENT = "0 iterations (no IBT)/"
LOG_DIR = f"./logs/{EXPERIMENT}"

SRC_LANG = "AAVE"
TGT_LANG = "SAE"

METRIC = evaluate.load("sacrebleu")

DATA_COLLATOR = DataCollatorForSeq2Seq(TOKENIZER, model=SOURCE_TO_TARGET_MODEL)

# SOURCE_TO_TARGET_MODEL.generation_config.max_new_tokens = 30
# TARGET_TO_SOURCE_MODEL.generation_config.max_new_tokens = 30


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
    # TODO: put data loading in a function
    source_to_target_data = parallel_data
    target_to_source_data = parallel_data

    # TODO: add a max_sequence_length variable to use for setting max_new_tokens
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

    # prepare monolingual data

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
    print(f"Training {target_lang} to {source_lang} model")

    # TODO: log predicted sequences in eval loop
    # TODO: add max_new_tokens for eval loop
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
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = TOKENIZER.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, TOKENIZER.pad_token_id)
    decoded_labels = TOKENIZER.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
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
    # used for paired txt files
    #
    # paired_src_data_path = f"/content/gdrive/MyDrive/6.861 Project/data/AAVE-SAE-data/{src_lang}_samples.txt"
    # paired_tgt_data_path = f"/content/gdrive/MyDrive/6.861 Project/data/AAVE-SAE-data/{tgt_lang}_samples.txt"
    #
    # raw_paired_dataset = Dataset.from_generator(
    #     yield_paired_lines,
    #     gen_kwargs={
    #         "source_path": paired_src_data_path,
    #         "target_path": paired_tgt_data_path,
    #         "source_lang": src_lang,
    #         "target_lang": tgt_lang,
    #     },
    # )

    # paired_csv_data_path = "/content/gdrive/MyDrive/6.861 Project/data/AAVE-SAE-data/GPT Translated AAVE Lyrics.csv"
    paired_csv_data_path = "/Users/willreed/projects/classes/nlp-final-project/GPT-Translated-AAVE-Lyrics.csv"

    raw_paired_dataset = Dataset.from_generator(
        yield_csv_lines,
        gen_kwargs={
            "csv_dataset_path": paired_csv_data_path,
            "source_lang": SRC_LANG,
            "target_lang": TGT_LANG,
            # use n for debugging
            # only loads n samples
            # "n": 1000,
        },
    )

    size_paired_dataset = len(raw_paired_dataset)

    # monolingual_src_data_path = "/content/gdrive/MyDrive/6.861 Project/data/AAVE-SAE-data/combined_AAVE_data.txt"
    # monolingual_tgt_data_path = (
    #     "/content/gdrive/MyDrive/6.861 Project/data/AAVE-SAE-data/cleaned_BAWE.txt"
    # )
    monolingual_src_data_path = (
        "/Users/willreed/projects/classes/nlp-final-project/coraal_dataset.txt"
    )
    monolingual_tgt_data_path = (
        "/Users/willreed/projects/classes/nlp-final-project/cleaned_BAWE.txt"
    )

    ratio = 1

    raw_monolingual_src_data = Dataset.from_generator(
        yield_mono_lines,
        gen_kwargs={
            "path": monolingual_src_data_path,
            "lang": SRC_LANG,
            "n": ratio * size_paired_dataset,
        },
    )
    raw_monolingual_tgt_data = Dataset.from_generator(
        yield_mono_lines,
        gen_kwargs={
            "path": monolingual_tgt_data_path,
            "lang": TGT_LANG,
            "n": ratio * size_paired_dataset,
        },
    )

    iterative_back_translation(
        parallel_data=raw_paired_dataset,
        source_to_target_model=SOURCE_TO_TARGET_MODEL,
        target_to_source_model=TARGET_TO_SOURCE_MODEL,
        tokenizer=TOKENIZER,
        source_data=raw_monolingual_src_data,
        target_data=raw_monolingual_tgt_data,
        iterations=3,
        source_lang=SRC_LANG,
        target_lang=TGT_LANG,
        num_epochs=3,
        log_dir=LOG_DIR,
    )


if __name__ == "__main__":
    main()
