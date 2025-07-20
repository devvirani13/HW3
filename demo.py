import os
import ollama
from math import sqrt

# === CONFIGURATION ===
DATA_FILE = 'cat-facts.txt'
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# === Load and Chunk the Dataset ===
dataset = []
with open(DATA_FILE, 'r', encoding='utf-8') as file:
    dataset = [line.strip() for line in file if line.strip()]

# === Build the Vector Store ===
VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

for chunk in dataset:
    add_chunk_to_database(chunk)

# === Cosine Similarity ===
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

# === Retrieval Function ===
def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# === Chatbot Response Generation ===
def generate_response(input_query):
    retrieved_knowledge = retrieve(input_query)

    print('\n📚 Retrieved knowledge:')
    for chunk, similarity in retrieved_knowledge:
        print(f' - (similarity: {similarity:.2f}) {chunk}')

    instruction_prompt = f"""You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{chr(10).join([f' - {chunk}' for chunk, _ in retrieved_knowledge])}
"""

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )

    print('\n🤖 Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

# === Main ===
if __name__ == '__main__':
    while True:
        try:
            input_query = input('\n\n📝 Ask me a question (or type "exit"): ')
            if input_query.strip().lower() == 'exit':
                break
            generate_response(input_query)
        except KeyboardInterrupt:
            break
