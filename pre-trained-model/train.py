import evaluate
from datasets import Dataset, Sequence, Value, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    pytorch_utils,
)
from utils import *


def patched_isin_mps_friendly(elements, test_elements):
    print("test")
    if test_elements.ndim == 0:
        test_elements = test_elements.unsqueeze(0)
    return (
        elements.tile(test_elements.shape[0], 1)
        .eq(test_elements.unsqueeze(1))
        .sum(dim=0)
        .bool()
        .squeeze()
    )


pytorch_utils.isin_mps_friendly = patched_isin_mps_friendly


def train_model(
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

    Returns
    =======
        the two models, trained
    """
    # prepare data, consider putting this in a function
    source_to_target_data = parallel_data
    target_to_source_data = parallel_data

    # source_to_target_prefix = f"Translate {source_lang} to {target_lang}: "
    # target_to_source_prefix = f"Translate {target_lang} to {source_lang}: "

    # don't include prefixes
    # when training, the model will predict the prefix
    # huggingface probably has a way to fix this

    # source_to_target_data = source_to_target_data.map(lambda x: {source_lang: source_to_target_prefix + x[source_lang]})
    # target_to_source_data = target_to_source_data.map(lambda x: {target_lang: target_to_source_prefix + x[target_lang]})

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

    # source_to_target_data = source_to_target_data.remove_columns(
    #     source_lang
    # ).remove_columns(target_lang)
    # target_to_source_data = target_to_source_data.remove_columns(
    #     source_lang
    # ).remove_columns(target_lang)

    # source_data = source_data.map(lambda x: {source_lang: source_to_target_prefix + x[source_lang]})
    # target_data = target_data.map(lambda x: {target_lang: target_to_source_prefix + x[target_lang]})

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

    # source_data = source_data.remove_columns(source_lang)
    # target_data = target_data.remove_columns(target_lang)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=source_to_target_model)

    target_to_source_output_dir = log_dir + "target_to_source_models/iteration 0"
    target_to_source_log_dir = log_dir + "target_to_source/iteration 0"

    target_to_source_training_args = Seq2SeqTrainingArguments(
        output_dir=target_to_source_output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        # keep at 10 for initial training
        num_train_epochs=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=target_to_source_log_dir,
        logging_steps=500,
        predict_with_generate=True,
        # fp16=True,
    )

    combined_target_to_source_data = target_to_source_data.train_test_split(
        test_size=0.1
    )

    target_to_source_trainer = Seq2SeqTrainer(
        model=target_to_source_model,
        args=target_to_source_training_args,
        train_dataset=combined_target_to_source_data["train"],
        eval_dataset=combined_target_to_source_data["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Iteration: 0")
    print(f"Training {target_lang} to {source_lang} model")

    for entry in combined_target_to_source_data["test"]:
        print(entry["input_ids"])
        print(entry["labels"])
        print(entry["attention_mask"])
        print()

    target_to_source_trainer.train()

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

        print_random_decoded_entries(
            dataset=synthetic_source_to_target_data,
            tokenizer=tokenizer,
            iteration=iteration,
            source_lang=source_lang,
            target_lang=target_lang,
            log_dir=target_to_source_log_dir,
        )

        combined_source_to_target_data = concatenate_datasets(
            [source_to_target_data, synthetic_source_to_target_data]
        )

        combined_source_to_target_data = combined_source_to_target_data.map(
            fix_attention_mask, batched=True
        )

        # generate train/test split and start training
        combined_source_to_target_data = (
            combined_source_to_target_data.train_test_split(test_size=0.1)
        )

        source_to_target_output_dir = (
            log_dir + f"source_to_target_models/iteration {iteration}"
        )
        source_to_target_log_dir = log_dir + f"source_to_target/iteration {iteration}"

        source_to_target_training_args = Seq2SeqTrainingArguments(
            output_dir=source_to_target_output_dir,
            learning_rate=1e-4,
            per_device_train_batch_size=8,
            num_train_epochs=num_epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir=source_to_target_log_dir,
            logging_steps=500,
            predict_with_generate=True,
            # fp16=True,
        )

        source_to_target_trainer = Seq2SeqTrainer(
            model=source_to_target_model,
            args=source_to_target_training_args,
            train_dataset=combined_source_to_target_data["train"],
            eval_dataset=combined_source_to_target_data["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        print(f"Iteration: {iteration}")
        print(f"Training {source_lang} to {target_lang} model")

        source_to_target_trainer.train()

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

        print_random_decoded_entries(
            dataset=synthetic_target_to_source_data,
            tokenizer=tokenizer,
            iteration=iteration,
            source_lang=target_lang,
            target_lang=source_lang,
            log_dir=source_to_target_log_dir,
        )

        combined_target_to_source_data = concatenate_datasets(
            [target_to_source_data, synthetic_target_to_source_data]
        )

        combined_target_to_source_data = combined_target_to_source_data.map(
            fix_attention_mask, batched=True
        )

        combined_target_to_source_data = (
            combined_target_to_source_data.train_test_split(test_size=0.1)
        )

        target_to_source_output_dir = (
            log_dir + f"target_to_source_models/iteration {iteration}"
        )
        target_to_source_log_dir = log_dir + f"target_to_source/iteration {iteration}"

        target_to_source_training_args = Seq2SeqTrainingArguments(
            output_dir=target_to_source_output_dir,
            learning_rate=1e-4,
            per_device_train_batch_size=8,
            num_train_epochs=num_epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir=target_to_source_log_dir,
            logging_steps=500,
            predict_with_generate=True,
            # fp16=True,
        )

        target_to_source_trainer = Seq2SeqTrainer(
            model=target_to_source_model,
            args=target_to_source_training_args,
            train_dataset=combined_target_to_source_data["train"],
            eval_dataset=combined_target_to_source_data["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        print(f"Iteration: {iterations}")
        print(f"Training {target_lang} to {source_lang} model")

        target_to_source_trainer.train()

    return source_to_target_model, target_to_source_model


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    source_to_target_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
    target_to_source_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

    src_lang = "AAVE"
    tgt_lang = "SAE"

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
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            # use n for debugging
            # only loads n samples
            "n": 1000,
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
            "lang": src_lang,
            "n": ratio * size_paired_dataset,
        },
    )
    raw_monolingual_tgt_data = Dataset.from_generator(
        yield_mono_lines,
        gen_kwargs={
            "path": monolingual_tgt_data_path,
            "lang": tgt_lang,
            "n": ratio * size_paired_dataset,
        },
    )

    metric = evaluate.load("sacrebleu")

    experiment = "0 iterations (no IBT)/"
    log_dir = f"./logs/{experiment}"

    train_model(
        parallel_data=raw_paired_dataset,
        source_to_target_model=source_to_target_model,
        target_to_source_model=target_to_source_model,
        tokenizer=tokenizer,
        source_data=raw_monolingual_src_data,
        target_data=raw_monolingual_tgt_data,
        iterations=0,
        source_lang=src_lang,
        target_lang=tgt_lang,
        num_epochs=3,
        log_dir=log_dir,
    )
