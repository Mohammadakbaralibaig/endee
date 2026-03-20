import os
from typing import List
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def generate_answer(question: str, context_chunks: List[str]) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Check your .env file.")

    client = Groq(api_key=api_key)

    context = "\n\n---\n\n".join(
        [f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )

    prompt = f"""You are a helpful AI assistant for industrial safety documents.
Answer the user's question based ONLY on the provided context chunks.
If the answer is not found, say: "I couldn't find relevant information in the document."
Be concise and accurate.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()