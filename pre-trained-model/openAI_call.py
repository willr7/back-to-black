import openai
from openai import OpenAI
import openAI_call
import os
import pandas as pd

# gets the api key environment variable to be able to access the OpenAI API
# openai.api_key = os.environ.get("OPENAI_API", "")
# need to set the OPENAI_API_KEY first
    # run this in terminal Mac/Linux
        # export OPENAI_API_KEY="your_api_key_here"
    # run this in terminal Windows
        # setx OPENAI_API_KEY "your_api_key_here"

api_key = open("openAI_api_key.txt", "r").read()
os.environ["OPENAI_API_KEY"] = api_key

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

text = ""
target_lang = "Standard Academic English"
source_lang = "African American Vernacular"
website_url = "https://genius.com/Travis-scott-sicko-mode-lyrics"
prompt = f"Translate the {source_lang} rap lyrics from this website to {target_lang}: {website_url}"

# # chat_translation = openai.Completion.create(
# chat_translation = client.completions.create(
#     model="gpt-40-mini",
#     # model="text-davinci-003",
#     prompt=prompt,
#     max_tokens=500
# )

# print(chat_translation.choices[0].text)

# client = OpenAI()

try:
    completion = client.chat.completions.create(
        model="gpt-40-mini",
        # model="gpt-3.5-turbo",
        # model="text-embedding-3-large", # free model that allows 1,000,000 tokens per min
        messages=[
            {"role":"system", "content": "You are a helpful African American Vernacular English(AAVE) to Standard Academic English(SAE) translator"},
            {"role":"user", "content": prompt}
            #  "Translate the following AAVE lyrics where each line is separated by a semicolon to SAE lyrics: "}
        ]
    )
    print(completion.choices[0].message)

    translated_chat = pd.DataFrame({"translated_text": list(completion.choices[0].message)})
    translated_chat.to_csv("translated_travis_scott_song.csv")

except openai.RateLimitError as e:
    print("Oh no you exceeded the rate limit because you broke :(")
    print(f"Error message: {e}")


# need to pay for all the requests to the openai api 
# will look into perplexity api 