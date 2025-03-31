# https://weave-docs.wandb.ai/quickstart

import os

import weave
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


client = OpenAI()


# Weave will track the inputs, outputs and code of this function
@weave.op()
def extract_dinos(sentence: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """In JSON format extract a list of `dinosaurs`, with their `name`,
their `common_name`, and whether its `diet` is a herbivore or carnivore""",
            },
            {"role": "user", "content": sentence},
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


# Initialise the weave project
weave.init("jurassic-park")

sentence = """I watched as a Tyrannosaurus rex (T. rex) chased after a Triceratops (Trike), \
both carnivore and herbivore locked in an ancient dance. Meanwhile, a gentle giant \
Brachiosaurus (Brachi) calmly munched on treetops, blissfully unaware of the chaos below."""

result = extract_dinos(sentence)
print(result)
