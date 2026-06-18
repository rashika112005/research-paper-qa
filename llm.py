from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_answer(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""You are a helpful research assistant. 
Answer the user's question based ONLY on the context provided below.
If the answer is not in the context, say "I couldn't find this in the uploaded paper."

Context:
{context}

Question:
{question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content