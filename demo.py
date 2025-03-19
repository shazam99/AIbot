from transformers import pipeline
import torch

qa_model = pipeline("text2text-generation", model="google/flan-t5-large", device=0 if torch.cuda.is_available() else -1)

context = """
The Bharat Bill Payment System (BBPS) is an integrated bill payment system in India that offers interoperable and accessible bill payment service to customers through a network of agents, enabling multiple payment modes and providing instant confirmation of payment.
"""

question = "full form of bbps."

response = qa_model(f"question: {question} context: {context}", max_new_tokens=200)

print(response[0]['generated_text'])


