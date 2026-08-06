from openai import OpenAI
from dotenv import load_dotenv

from src.generation.prompt_builder import build_prompt

load_dotenv()

client = OpenAI()


class Generator:

    def generate(
        self,
        question: str,
        chunks: list[str],
        history: list[dict]
    ) -> str:

        prompt = build_prompt(
            question,
            chunks,
            history
        )

        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt
        )

        return response.output_text


    def stream_generate(
        self,
        question: str,
        chunks: list[str],
        history: list[dict]
    ):

        prompt = build_prompt(
            question,
            chunks,
            history
        )

        stream = client.responses.create(
            model="gpt-5-nano",
            input=prompt,
            stream=True
        )

        partial = ""

        for event in stream:

            if event.type == "response.output_text.delta":
                partial += event.delta
                yield partial