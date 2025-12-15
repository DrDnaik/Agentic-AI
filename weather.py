from langchain.agents import Tool, initialize_agent
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import requests


load_dotenv() # env
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini',temperature=0.2)


def get_weather(city):

    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    geo_data = requests.get(geo_url).json()
    latitude = geo_data['results'][0]['latitude']
    longitude = geo_data['results'][0]['longitude']
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    weather_data = requests.get(weather_url).json()
    temparature = weather_data['current_weather']['temperature']
    windspeed = weather_data['current_weather']['windspeed']
    return f"Current temparature in {city} is {temparature}°C and the wind speed is {windspeed} km/hr"

# Tool defining

tools = [
    Tool(
        name = "weather_fetcher",
        func = get_weather,
        description = "Use this tool ONLY when the user specifically asks about weather or temperature or climate for a specific city. Input should be a city name."
    )
]


# Create an Agent

'''
Gen AI : Able to generate answers. No actions are involved.

Agentic AI : Agents which will think and act like humans on its own.

What is AI ?  --> (Think : I am capable of answering this qs Reason :  I dont need any help. Answer :  )
What is the weather in Paris ? --> (Think : I can't , Tool - weather tool , get the answer, LLM to answer. )

'''

agent = initialize_agent(
    tools,
    llm,
    agent = "zero-shot-react-description",
    verbose = True
)

user_query = input('Enter a query : ')

response = agent.invoke({'input':user_query})
print(response['output'])


'''
What is the weather in Paris ? 


1. I need the help from Tool 
2. It needs to extract the city from the input 
3. Call the tool, get the weather
4. Send the answer to LLM, and give the final answer

'''
# from langgraph.prebuilt import create_react_agent
#
# agent = create_react_agent(llm, tools)






