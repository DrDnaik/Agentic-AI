from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import requests

# Load API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY")

# Weather function
def get_weather(city):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    geo_data = requests.get(geo_url).json()

    latitude = geo_data['results'][0]['latitude']
    longitude = geo_data['results'][0]['longitude']

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}&current_weather=true"
    )
    weather_data = requests.get(weather_url).json()

    temp = weather_data['current_weather']['temperature']
    wind = weather_data['current_weather']['windspeed']

    return f"Current temperature in {city} is {temp}°C and wind speed is {wind} km/hr."

# LLM with strict weather-only rule
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

SYSTEM_INSTRUCTIONS = """
You are a STRICT weather assistant.

RULES:
1. You ONLY answer weather-related questions.
2. If the user asks about weather, temperature, climate, humidity, wind, or forecast for a city:
   - Respond ONLY with: CITY=<cityname>
   (Example: CITY=London)
3. For ANY non-weather question, reply EXACTLY:
   "I am a weather app."
4. DO NOT answer anything else.
"""

def think(query):
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": query}
    ])
    return response.content

# Main logic
user_query = input("Enter a query: ")

llm_decision = think(user_query)

if llm_decision.startswith("CITY="):
    city = llm_decision.replace("CITY=", "").strip()
    print(get_weather(city))
else:
    print(llm_decision)
