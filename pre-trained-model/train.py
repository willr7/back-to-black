from transformers import TrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_data():
    return


def train_model(parallel_data, source_to_target_model, target_to_source_model, source_data, target_data, tokenizer, iterations):
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
        # keep in mind memory concerns, use yield if possible
        synthetic_source_data = translate(target_to_source_model, target_data)

        # combine() should combine the synthetic data and the target data, and then add them to the parallel dataset
        source_to_target_data = combine(synthetic_source_data, target_data, parallel_data)

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

        target_to_source_data = combine(synthetic_target_data, source_data, parallel_data)


    return source_to_target_model, target_to_source_model


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
