Retrieval-Augmented Generation (RAG) System with Ollama

This is a simple project where I implemented a Retrieval-Augmented Generation (RAG) system using Python and the Ollama API for running local language models. The idea is to use document retrieval (based on embeddings) to find the most relevant facts, and then generate answers using a local LLM (LLaMA 3.2-1B Instruct).

What the Code Does

1. Loads a dataset of short factual lines from a text file (like 'cat-facts.txt').
2. Uses an embedding model (bge-base-en) to generate embeddings for each line.
3. Stores these embeddings in a list called VECTOR_DB.
4. When the user asks a question, it:
   - Embeds the query
   - Finds the most similar chunks using cosine similarity
   - Builds a prompt from the retrieved chunks
   - Uses a local LLM to generate a response based only on that context
5. The chatbot response is streamed in real time.

How to Run

Requirements:
- Python 3.8 or higher
- Ollama installed from https://ollama.com
- Download the models by running these commands:
    ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
    ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
- Install the Python package:
    pip install ollama

Steps:
1. Make sure you have a file called cat-facts.txt in the same folder (each line = one fact).
2. Run the script:
    python demo.py
3. Type your questions in the terminal. Type 'exit' to quit.

Task 8: Experimentation & Reflection

1. Trying different top_n values:
   - top_n = 1: Very focused, but sometimes misses additional useful context.
   - top_n = 3: Default and balanced. Includes diverse and relevant facts.
   - top_n = 5: Includes more info, but may be repetitive or slightly off-topic.

2. What limitations did you observe?
   - The system depends a lot on the way the facts and questions are written.
   - With a small dataset, answers are often repetitive or too simple.
   - It can’t combine facts or reason very well.
   - Sometimes the model still makes up information if it doesn’t find good context.

3. What could be improved with a larger dataset or better models?
   - Using larger, paragraph-based datasets would give more meaningful answers.
   - A stronger LLM would generate more accurate and fluent responses.
   - Combining sparse (BM25) and dense retrieval could help improve result quality.
   - Adding metadata or chunking full documents might give better grounding and traceability.

Files Included

- demo.py         : The full working script
- cat-facts.txt   : Sample dataset (can be replaced with your own)
- README.md       : This documentation

