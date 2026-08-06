from src.generation.generator import Generator

generator = Generator()

chunks = [
    "Machine learning is the science of getting computers to learn without being explicitly programmed."
]

question = "What is machine learning?"

answer = generator.generate(
    question,
    chunks
)

print(answer)