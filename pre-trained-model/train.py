from datasets import Dataset, load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer,
                          TrainingArguments)


def train_model(
    parallel_data,
    source_to_target_model,
    target_to_source_model,
    tokenizer,
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
        synthetic_source_data = translate(target_to_source_model, target_data, tokenizer)

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

        synthetic_target_data = translate(source_to_target_model, source_data, tokenizer)

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

def translate(translation_model, monolingual_data, tokenizer):
    """
    Creates synthetic data using a translation model and monolingual dataset

    Parameters
    ==========
    translation_model: torch.tensor
        pretrained model that can translate source data into target data

    monolingual_data: torch.

    Returns
    =======
    """
    # might need to prefix each line of the data with f'translate {source_language} to {target_language}'
    # assume the data does not have translation prompt prefix
    # what data type is the monolingual data, dictionary? list? HuggingFace Dataset?

    # prefix = f'translate {source_lang} to {target_lang}: '
    # new_mono_data = [prefix + text for text in monolingual_data]
    input_tokens = tokenizer(monolingual_data, return_tensors='pt').input_ids

    # translates the tokens from the source language to tokens in the target language
    output_tokens = translation_model.generate(input_tokens, max_new_tokens=40, top_k=10, top_p=0.95)
    # decode the generated token ids back into text from the target language
    synthetic_data = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return synthetic_data

def combine(source_data, synthetic_target_data, parallel_data, source_lang, target_lang):
    """
    Creates a new HuggingFace Dataset with the previous parallel data and adds on 
    the paired source and synthetic target data

    Paramter
    ========
    source_data: dict, source_lang: source_text
        dictionary of sentences from the source language

    synthetic_target_data: dict, target_lang: target_text
        dictionary of sentences from the target language

    parallel_data: dict, source_lang: source_text, target_lang: target_lang
        HuggingFace dictionary of paired sentences from the source and target languages

    Returns
    ======= 
        HuggingFace dataset with the keys, values items source_lang: source_text, target_lang: target_text
    """
    # mono_dataset => {lang: line}
    # paired_dataset => {source_lang: source_line, target_lang: target_line}
    
    # can't append to a HuggingFace Dataset, immutable (uses similar interface to an array), 
    # have to create a new dataset object 
    
    assert len(synthetic_target_data) == len(source_data)

    source_lines = [source_dict[source_lang] for source_dict in source_data]
    target_lines = [target_dict[target_lang] for target_dict in synthetic_target_data]

    for paired_dict in parallel_data:
        source_text = paired_dict[source_lang]
        target_text = paired_dict[target_lang]

        source_lines.append(source_text)
        target_lines.append(target_text)

    parallel_dictionary = {source_lang: source_lines, target_lang: target_lines}
    combined_parallel_data = Dataset.from_dict(parallel_dictionary)

    return combined_parallel_data

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

