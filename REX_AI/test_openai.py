from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv(override=True)

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.create(
        model="gpt-4.1-mini",
        input="Reply with exactly: OpenAI connection successful."
    )

    print("SUCCESS:", response.output_text)

except Exception as e:
    print("ERROR:", type(e).__name__)
    print("DETAILS:", str(e))