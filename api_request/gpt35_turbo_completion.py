import openai
from config import CONFIG

"""GPT3.5 completion"""

def gpt35_turbo_completion(prompt_text):
    openai.api_key = CONFIG['openai_api_key']
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt_text}
    ],
    # max_tokens=200,
    temperature=0,
    # stop=['--', '\n', ';'],
    )
    return completion.choices[0].message["content"]
