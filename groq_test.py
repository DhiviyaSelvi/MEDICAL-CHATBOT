from groq import Groq
from dotenv import load_dotenv
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chat = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Explain fever in simple medical terms"}
    ],
    model="openai/gpt-oss-20b"

)

print(chat.choices[0].message.content)
