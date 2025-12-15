from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
apikey= os.getenv("OPEN_API_KEY")
client= OpenAI(api_key=apikey)
response= client.responses.create(
    model= 'gpt-4o-mini',
    input= 'how do a surgery'
)
print(response.output_text)


