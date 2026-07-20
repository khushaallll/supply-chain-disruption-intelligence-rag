# example for Groq/OpenAI-compatible endpoint
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import os
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
resp = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "say ok"}]
)
print(resp.choices[0].message.content)

model = SentenceTransformer("intfloat/e5-large-v2")
emb = model.encode("test sentence")
print(emb.shape)  # should be (1024,)