from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def construct_sentence(word_list):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        organization="org-rx9x4YOvF5JGUb2Brq8DhIl8"
    )

    openai_instructions = '''
    You are an AI designed to assist deaf content creators by generating captions for their videos.
    [Your characteristics]
    1. These videos feature sign language, and the goal is to create accurate and coherent captions to ease their content creation process
    2. Your task is to construct proper and LOGICAL sentences from a list of words provided, in which the input may or may not be in order.
    3. Each segment should be a complete sentence. Predictions should be kept to a minimum to ensure accuracy, reflecting the exact sign language input as closely as possible.
    Here are the words:
    '''

    input_text = " ".join(word_list)

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": f'{openai_instructions}'},
            {"role": "user", "content": f'{input_text}'}
        ]
    )

    generated_sentence = response.choices[0].message.content

    return generated_sentence
