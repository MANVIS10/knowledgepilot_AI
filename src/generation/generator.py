from openai import OpenAI
from dotenv import load_dotenv

from src.generation.prompt_builder import build_prompt
from src.config import MODEL_NAME

load_dotenv()

client = OpenAI()


class Generator:

    def generate(
        self,
        question: str,
        chunks: list,
        history: list[dict] = None
    ) -> str:

        if history is None:
            history = []

        prompt = build_prompt(
            question,
            chunks,
            history
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content


    def stream_generate(
        self,
        question: str,
        chunks: list,
        history: list[dict] = None
    ):

        if history is None:
            history = []

        prompt = build_prompt(
            question,
            chunks,
            history
        )


        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        partial = ""

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                partial += chunk.choices[0].delta.content
                yield partial