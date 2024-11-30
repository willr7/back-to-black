from datasets import Dataset, load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np


def train_model(
    parallel_data,
    source_to_target_model,
    target_to_source_model,
    tokenizer,
    source_data,
    target_data,
    iterations,
    source_lang,
    target_lang,
):
    # prepare data, consider putting this in a function
    source_to_target_data = parallel_data
    target_to_source_data = parallel_data

    source_to_target_prefix = f"Translate {source_lang} to {target_lang}: "
    target_to_source_prefix = f"Translate {target_lang} to {source_lang}: "

    source_to_target_data = source_to_target_data.map(lambda x: {source_lang: source_to_target_prefix + x[source_lang]})
    target_to_source_data = target_to_source_data.map(lambda x: {target_lang: target_to_source_prefix + x[target_lang]})

    source_to_target_data = source_to_target_data.map(
        preprocess_source_to_target_function,
        batched=True,
    )
    target_to_source_data = source_to_target_data.map(
        preprocess_target_to_source_function,
        batched=True,
    )

    source_to_target_data = source_to_target_data.remove_columns("AAVE").remove_columns("SAE")
    target_to_source_data = target_to_source_data.remove_columns("AAVE").remove_columns("SAE")

    source_to_target_data = source_to_target_data.train_test_split(test_size=0.1)
    target_to_source_data = target_to_source_data.train_test_split(test_size=0.1)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=source_to_target_model)

    source_to_target_training_args = Seq2SeqTrainingArguments(
        output_dir="./source_to_target_models",
        eval_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        num_train_epochs=5,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=500,
        predict_with_generate=True,
        fp16=True,
    )

    target_to_source_training_args = Seq2SeqTrainingArguments(
        output_dir="./target_to_source_models",
        eval_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        num_train_epochs=5,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=500,
        predict_with_generate=True,
        fp16=True,
    )


    for _ in range(iterations):
        target_to_source_trainer = Seq2SeqTrainer(
            model=target_to_source_model,
            args=target_to_source_training_args,
            train_dataset=target_to_source_data["train"],
            eval_dataset=target_to_source_data["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        target_to_source_trainer.train()

        synthetic_source_data = translate(
            target_to_source_model, target_data, tokenizer, source_lang, target_lang
        )

        source_to_target_data = combine(
            synthetic_source_data, target_data, parallel_data, source_lang, target_lang
        )

        source_to_target_trainer = Seq2SeqTrainer(
            model=source_to_target_model,
            args=source_to_target_training_args,
            train_dataset=source_to_target_data["train"],
            eval_dataset=source_to_target_data["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        source_to_target_trainer.train()

        synthetic_target_data = translate(
            target_to_source_model, target_data, tokenizer, source_lang, target_lang
        )

        target_to_source_data = combine(
            source_data, synthetic_target_data, parallel_data, source_lang, target_lang
        )

    return source_to_target_model, target_to_source_model


def preprocess_source_to_target_function(examples, max_length=200):
    inputs = examples[src_lang] 
    targets = examples[tgt_lang]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs

def preprocess_target_to_source_function(examples, max_length=200):
    inputs = examples[tgt_lang] 
    targets = examples[src_lang]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs

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


def translate(translation_model, monolingual_data, tokenizer, source_lang, target_lang):
    """
    Creates synthetic data using a translation model and monolingual dataset

    Parameters
    ==========
    translation_model: Dataset
        pretrained model that can translate sentences from source language into sentences from target language

    monolingual_data: Dataset
        list of dictionaries with item {lang: text}

    tokenizer: torch tokenizer?
        object that can tokenize list of text

    source_lang: str
        the source language

    target_lang: str
        the target language

    Returns
    =======
        returns a HuggingFace Dataset using the following dictionary {lang: [lines of translated text]}
    """
    # might need to prefix each line of the data with f'translate {source_language} to {target_language}'
    # assume the data does not have translation prompt prefix
    # what data type is the monolingual data, dictionary? list? HuggingFace Dataset?

    prompt = f'translate {source_lang} to {target_lang}: '
    new_mono_data = [prompt + source_dict[source_lang] for source_dict in monolingual_data]
    synthetic_lines = []
    
    for text in new_mono_data:
        # tokenizer just takes in the lines from the language
        input_token = tokenizer(text, return_tensors="pt").input_ids

        # translates the tokens from the source language to tokens in the target language
        # top 5 tokens retained
        output_token = translation_model.generate(
            input_token, max_new_tokens=40, top_k=5, top_p=0.95
        )
        # decode the generated token ids back into text from the target language
        # not sure if I should only index first element, might change this code later once we get data
        synthetic_line = tokenizer.decode(output_token[0], skip_special_tokens=True)
        synthetic_lines.append(synthetic_line)
    synthetic_data = Dataset.from_dict({target_lang: synthetic_lines})

    return synthetic_data


def combine(source_data, target_data, parallel_data, source_lang, target_lang):
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
    # I assumed the data will look like the following:
    # source_dataset => {lang: line}
    # synthetic_target_data => {lang: line}
    # parallel_dataset => {source_lang: source_line, target_lang: target_line}

    # can't append to a HuggingFace Dataset, immutable (uses similar interface to an array),
    # have to create a new dataset object

    assert len(target_data) == len(source_data)

    source_lines = [source_dict[source_lang] for source_dict in source_data]
    target_lines = [target_dict[target_lang] for target_dict in target_data]

    for paired_dict in parallel_data:
        source_text = paired_dict[source_lang]
        target_text = paired_dict[target_lang]

        source_lines.append(source_text)
        target_lines.append(target_text)

    parallel_dictionary = {source_lang: source_lines, target_lang: target_lines}
    combined_parallel_data = Dataset.from_dict(parallel_dictionary)

    return combined_parallel_data

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    source_to_target_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
    target_to_source_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

    src_lang = "AAVE"
    tgt_lang = "SAE"

    paired_src_data_path = f"nlp-final-project/src/data/{src_lang}_samples.txt"
    paired_tgt_data_path = f"nlp-final-project/src/data/{tgt_lang}_samples.txt"

    raw_paired_dataset = Dataset.from_generator(
        yield_paired_lines,
        gen_kwargs={
            "source_path": paired_src_data_path,
            "target_path": paired_tgt_data_path,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
        },
    )

    monolingual_src_data_path = "coraal_dataset.txt"
    monolingual_tgt_data_path = "cleaned_BAWE.txt"

    raw_monolingual_src_data = Dataset.from_generator(
        yield_mono_lines,
        gen_kwargs={"path": monolingual_src_data_path, "lang": src_lang},
    )
    raw_monolingual_tgt_data = Dataset.from_generator(
        yield_mono_lines,
        gen_kwargs={"path": monolingual_tgt_data_path, "lang": tgt_lang},
    )
    
    raw_monolingual_tgt_data = raw_monolingual_tgt_data.filter(lambda x: len(x['SAE']) > 50 and len(x['SAE']) < 65)

    metric = evaluate.load("sacrebleu")

    train_model(raw_paired_dataset, source_to_target_model, target_to_source_model, tokenizer, raw_monolingual_src_data, raw_monolingual_tgt_data, 1, src_lang, tgt_lang)
