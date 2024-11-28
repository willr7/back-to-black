from datasets import Dataset, load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer,
                          TrainingArguments)


def train_model(
    parallel_data,
    source_to_target_model,
    target_to_source_model,
    source_data,
    target_data,
    iterations,
):
    source_to_target_data = parallel_data
    target_to_source_data = parallel_data

    source_to_target_training_args = TrainingArguments(
        output_dir="./source_to_target_models",
        evaluation_strategy="steps",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=500,
    )

    target_to_source_training_args = TrainingArguments(
        output_dir="./target_to_source_models",
        evaluation_strategy="steps",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=500,
    )

    target_to_source_trainer = Seq2SeqTrainer(
        model=target_to_source_model,
        args=target_to_source_training_args,
        train_dataset=target_to_source_data,
        eval_dataset=target_to_source_data,
    )

    for _ in range(iterations):
        target_to_source_trainer.train()
        # translate should return a huggingface dataset with the key "source"
        synthetic_source_data = translate(target_to_source_model, target_data)

        # combine() should combine the synthetic data and the target data, and then add them to the parallel dataset
        source_to_target_data = combine(
            synthetic_source_data, target_data, parallel_data
        )

        # TODO
        # consider adding a grammar checker loss after testing iterative back translation
        source_to_target_trainer = Seq2SeqTrainer(
            model=source_to_target_model,
            args=source_to_target_training_args,
            train_dataset=source_to_target_data,
            eval_dataset=source_to_target_data,
        )

        source_to_target_trainer.train()

        synthetic_target_data = translate(source_to_target_model, source_data)

        target_to_source_data = combine(
            source_data, synthetic_target_data, parallel_data
        )

    return source_to_target_model, target_to_source_model


def yield_paired_lines(source_path, target_path, source_lang, target_lang):
    with open(source_path, "r", encoding="utf-8") as source_text_file, open(
        target_path, "r", encoding="utf-8"
    ) as target_text_file:
        for source_line, target_line in zip(source_text_file, target_text_file):
            yield {source_lang: source_line, target_lang: target_line}


def yield_mono_lines(path, lang):
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            yield {lang: line}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    source_to_target_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
    target_to_source_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

    paired_src_data_path = ""
    paired_tgt_data_path = ""

    src_lang = "AAVE"
    tgt_lang = "SAE"

    raw_paired_dataset = Dataset.from_generator(
        yield_paired_lines,
        gen_kwargs={
            "source_path": paired_src_data_path,
            "target_path": paired_tgt_data_path,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
        },
    )

    monolingual_src_data_path = ""
    monolingual_tgt_data_path = ""

    raw_monolingual_src_data = Dataset.from_generator(
        yield_mono_lines,
        gen_kwargs={"path": monolingual_src_data_path, "lang": src_lang},
    )
    raw_monolingual_tgt_data = Dataset.from_generator(
        yield_mono_lines,
        gen_kwargs={"path": monolingual_tgt_data_path, "lang": tgt_lang},
    )

