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
