import glob
import os
import re

import openai
import openAI_call
import pandas as pd
from openai import OpenAI


client = OpenAI()


def yield_source_data_lines(source_data_folder_path):
    source_data_file_paths = glob.glob(os.path.join(source_data_folder_path, "*.txt"))

    for file_path in source_data_file_paths:
        with open(file_path, "r") as f:
            for line in f.readlines():
                yield line


def load_source_data(source_data_folder_path, destination_path):
    source_data = []

    patterns_to_exclude = [
        r"^Title:.*",  # Lines starting with 'Title:'
        r"^Artist:.*",  # Lines starting with 'Artist:'
        r"^Album:.*",  # Lines starting with 'Album:'
        r"^Lyrics:.*",  # Lines starting with 'Lyrics:'
        r"^Song Genius Url:.*",  # Lines starting with 'Song Genius Url:'
        r"^Song Date:.*",  # Lines starting with 'Song Date:'
        r"^\[.*\]$",  # Lines enclosed in brackets (e.g., [Intro: Adrian Marcel])
    ]

    combined_pattern = re.compile("|".join(patterns_to_exclude))

    for line in yield_source_data_lines(source_data_folder_path):
        if not combined_pattern.match(line) and len(line.split()) >= 4:
            print(line)
            source_data.append(line)

    with open(destination_path, "a") as f:
        f.writelines(source_data)


def generate_translations(source_data_lines, destination_path):
    target_lang = "Standard Academic English"
    source_lang = "African American Vernacular"

    translated_pairs = {
        source_lang: [],
        target_lang: [],
    }
    
    batch_lines = 20

    for i in range(0, len(source_data_lines), batch_lines):
        input_lines = source_data_lines[i : i + batch_lines]
        input_string = "".join(input_lines)
        prompt = (
            f"Translate the following {source_lang} sample to {target_lang}:\n{input_string}"
        )
        print(i)
        print()
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an African American Vernacular English to Standard Academic English translator. Translate each line individually and separate them with a new line character, but don't add punctuation if it is not already there. Ensure that the number of lines in the output matches the number of lines in the given African American Vernacular sample.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            prediction = completion.choices[0].message.content 

            print(input_string)
            print(prediction)
            print()

            prediction_lines = prediction.split("\n")

            if len(prediction_lines) == batch_lines:
                translated_pairs[source_lang] += (input_lines)
                translated_pairs[target_lang] += (prediction_lines)

        except openai.RateLimitError as e:
            print("Oh no you exceeded the rate limit because you broke :(")
            print(f"Error message: {e}")


    pd.DataFrame(translated_pairs).to_csv(destination_path, index=False)

if __name__ == "__main__":

    # source_data_folder_path = "/Users/willreed/Downloads/Boosie Song Lyrics"
    # this file currently includes Common and Boosie
    source_data_destination_path = "/Users/willreed/nlp-final-project/AAVE Lyrics.txt"
    #
    # load_source_data(source_data_folder_path, source_data_destination_path)
    synthetic_target_destination_path = "/Users/willreed/nlp-final-project/GPT Translated AAVE Lyrics.csv"

    source_data_lines = []

    with open(source_data_destination_path, "r") as f:
        source_data_lines = f.readlines()

    generate_translations(source_data_lines, synthetic_target_destination_path)
