from openai import OpenAI
from config.env import openai_instructions

def construct_sentence(word_list):
    client = OpenAI()

    input_text = " ".join(word_list)

    response = client.Completion.create(
        model="gpt-4o",
        prompt=f"{openai_instructions}: {input_text}",
        max_tokens=30,
        temperature=0.7,
    )

    sentence = response.choices[0].text.strip()

    return sentence