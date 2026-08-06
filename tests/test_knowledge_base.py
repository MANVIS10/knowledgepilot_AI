from src.pipeline.knowledge_base import KnowledgeBase


kb = KnowledgeBase()

question = "What is machine learning?"

answer = kb.ask(question)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer)