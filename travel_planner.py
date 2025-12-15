
'''

1. Master Travel Agent : 

Tools : 

--> Weather Tool  [ Getting the real time weather ] : No LLM 

--> Weather and Season Advisor Agent : (2025, Nov : Current Month --> Weather Tool , General weather advice )

--> Transport Route Planner Agent  [ Flights , trains...]   [ Flight API ] : No LLM

--> Accomodation Finder Agent 

--> Activities and Attractions Curator Agent

--> BUdget Optimizer 

--> Safety and Local Tips Provider 

--> Travel Documents : Visa / Passport

'''

import os
from typing import TypedDict, List
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import Tool, initialize_agent, AgentType
import os
from dotenv import load_dotenv
import requests
from datetime import datetime

load_dotenv()  # env
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)

agent_results = {}


# -------------- Real Time Weather Tool ------------------

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


# -------------- Agent 1 : Weather and Season Advisor Agent -----------

'''
1. Month 
2. Destination

--> Pondicherry , December   [General]
--> If the travel month == Current Month --> Current Weather 
'''


def get_weather_advice(destination_info):
    city = destination_info.split('Destination:')[1].split(',')[0].strip()
    travel_month = destination_info.split('Travel Month:')[1].strip()
    current_month = datetime.now().strftime('%B')

    current_weather_info = ''

    if travel_month.lower() == current_month.lower():
        current_weather = get_weather(city)
        current_weather_info = f"\n\n Current Weather Data (Real-time) : \n {current_weather} \n\n Note : Since you're planning to travel this month, use the real-time weather data above for accurate current conditions"

    prompt = f""" You are a Weather and Season Advisor for travel planning. 

    {destination_info}{current_weather_info}

    Provide : 

    1. Expected weather conditions {' (use the real-time data provided above since travel is planned for the current month)' if current_weather_info else 'for the specified travel month'}
    2. Best time to visit the destination
    3. Packing recommendations
    4. Sessional events or festivals
    5. Any weather-related warnings

    Be specific and practical
    """

    response = llm.invoke(prompt)
    result = response.content
    agent_results['weather'] = result
    return result


# -------------- Agent 2 :  Transport Route Planner Agent -----------

def plan_transport_routes(trip_info):
    prompt = f""" You are a Transport ROute Planner.

    {trip_info}

     Provide:
    1. Best way to reach the destination (flight/train/bus)
    2. Local transportation options within destination
    3. Cost comparison: economy vs comfort options
    4. Travel time estimates
    5. Booking tips and recommendations

    Be practical and budget-conscious."""

    response = llm.invoke(prompt)
    result = response.content
    agent_results['transport'] = result
    return result


# -------------- Agent 3 :  Accomodation Finder Agent  -----------

def find_accomodations(accommodation_info):
    prompt = f"""You are an Accommodation Finder specialist.

    {accommodation_info}

    Recommend:
    1. Types of accommodation (hotel/hostel/resort/homestay)
    2. Best areas to stay with reasoning
    3. Price range per night
    4. Key amenities to look for
    5. Booking platforms to use
    6. 3 specific accommodation suggestions with pros/cons

    Match recommendations to budget and group type."""

    response = llm.invoke(prompt)
    result = response.content
    agent_results['accomodation'] = result
    return result


# -------------- Agent 4 :  Activities and Attractions Curator Agent  -----------


def curate_activites(acitivity_info):
    prompt = f"""You are an Activities & Attractions Curator.

    {acitivity_info}

    Create a day-wise activity plan:
    1. Must-visit attractions with timings
    2. Local food experiences and restaurant recommendations
    3. Free activities and hidden gems
    4. Cultural experiences
    5. Adventure activities (if applicable)
    6. Shopping areas
    7. Evening/nightlife options

    Organize by day and balance paid vs free activities."""

    response = llm.invoke(prompt)
    result = response.content
    agent_results['activities'] = result
    return result


# -------------- Agent 5 : BUdget Optimizer  Agent  -----------

def optimize_budget(budget_info):
    prompt = f"""You are a Budget Optimization specialist.

    {budget_info}

    Provide detailed budget breakdown:
    1. Transportation costs (to/from + local)
    2. Accommodation costs (per night × total nights)
    3. Food & dining (breakfast, lunch, dinner, snacks)
    4. Activities & entrance fees
    5. Shopping & miscellaneous
    6. Emergency buffer (10-15%)

    Total estimated cost:
    - Budget option
    - Mid-range option
    - Comfortable option

    Include money-saving tips and what to splurge on."""

    response = llm.invoke(prompt)
    result = response.content
    agent_results['budget'] = result
    return result


# -------------- Agent 6 : Safety and Local Tips Provider Agent -----

def provide_safety_tips(saftey_info):
    prompt = f"""You are a Safety & Local Tips advisor.

    {saftey_info}

    Provide:
    1. General safety tips for the destination
    2. Areas to avoid
    3. Local transportation safety
    4. Emergency contact numbers
    5. Cultural etiquette and do's/don'ts
    6. Common scams to watch out for
    7. Language tips or useful phrases
    8. Tipping customs
    9. Local laws to be aware of

    Be specific and helpful."""

    response = llm.invoke(prompt)
    result = response.content
    agent_results['safety'] = result
    return result


# -------------- Agent 7 : Docs Agent -----

def check_travel_documents(document_info):
    prompt = f"""You are a Travel Document & Visa Assistant.

{document_info}

Provide information on:
1. Visa requirements (visa-free/visa-on-arrival/e-visa/embassy visa)
2. Passport validity requirements
3. Required documents for visa application
4. Processing time and costs
5. Travel insurance recommendations
6. Vaccination requirements
7. Health certificates needed
8. Customs regulations

Be clear about mandatory vs optional requirements."""

    response = llm.invoke(prompt)
    result = response.content
    agent_results['documents'] = result
    return result


# -------------- Agent 8 : Generate Final Itinerary Agent -----

def generate_final_itinerary(complete_info):
    prompt = f"""You are a Final Itinerary Generator. Create a comprehensive, well-formatted travel itinerary.

    {complete_info}

    Create a COMPLETE itinerary with : 

    1. Trip Overview (destination, dates, duration, travelers)
    2. Day-by-Day Schedule
    - Morning activities
    - Afternoon activities
    - Evening activities
    - Meal suggestions
    - Transportation needed
    3. Pre-Trip Checklist
    4. Budget Summary
    5. Important Contacts & Tips
    6. Packing List Highlights

    Format it clearly with headers and bullet points.
    Make it balanced - not too rushed, not too relaxed.
    """

    response = llm.invoke(prompt)
    return response.content


# Master Agent :

def create_master_travel_agent(trip_data):
    tools = [
        Tool(
            name="Current Weather Fetcher",
            func=get_weather,
            description="Use this tool ONLY to get current real-time weather (temperature and wind speed) for a specific city. Input should be a city name."
        ),
        Tool(
            name="Weather and Season Advisor",
            func=lambda x: get_weather_advice(
                f"Destination: {trip_data['destination']}, Travel Month: {trip_data['travel_month']}"),
            description="Use this tool to get comprehensive weather analysis, seasonal information, packing recommendations, and festivals for the destination. This includes current weather data."
        ),
        Tool(
            name="Transport Route Planner",
            func=lambda x: plan_transport_routes(
                f"Destination: {trip_data['destination']}, Duration: {trip_data['days']} days, Budget: {trip_data['budget']}"),
            description="Use this tool to plan transportation including flights, trains, buses, and local transport options."
        ),
        Tool(
            name="Accommodation Finder",
            func=lambda x: find_accomodations(
                f"Destination: {trip_data['destination']}, Budget: {trip_data['budget']}, Group Type: {trip_data['group_type']}, Duration: {trip_data['days']} days"),
            description="Use this tool to find suitable hotels, hostels, resorts, or homestays based on budget and preferences."
        ),
        Tool(
            name="Activities and Attractions Curator",
            func=lambda x: curate_activites(
                f"Destination: {trip_data['destination']}, Interests: {trip_data['interests']}, Duration: {trip_data['days']} days, Budget: {trip_data['budget']}"),
            description="Use this tool to get day-wise activity plans, must-visit attractions, food recommendations, and cultural experiences."
        ),
        Tool(
            name="Budget Optimizer",
            func=lambda x: optimize_budget(
                f"Destination: {trip_data['destination']}, Duration: {trip_data['days']} days, Budget: {trip_data['budget']}, Group Type: {trip_data['group_type']}"),
            description="Use this tool to get detailed budget breakdown and cost optimization strategies for the trip."
        ),
        Tool(
            name="Safety and Local Tips Provider",
            func=lambda x: provide_safety_tips(
                f"Destination: {trip_data['destination']}, Traveler Type: {trip_data['group_type']}"),
            description="Use this tool to get safety tips, local customs, cultural etiquette, and emergency information."
        ),
        Tool(
            name="Travel Documents and Visa Assistant",
            func=lambda x: check_travel_documents(
                f"Destination: {trip_data['destination']}, Nationality: {trip_data.get('nationality', 'Indian')}"),
            description="Use this tool to check visa requirements, passport validity, vaccination needs, and travel documents."
        )
    ]

    master_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    return master_agent


def run_travel_planning(user_requirements):
    global agent_results

    agent_results = {}

    master_agent = create_master_travel_agent(user_requirements)

    planning_prompt = f""" You are an intelligent travel planning assistant. help plan a trip based on these requirements : 

    ** Trip Details : **

    - Destination : {user_requirements['destination']}
    - Duration : {user_requirements['days']}
    - Travel Month : {user_requirements['travel_month']}
    - Budget Level : {user_requirements['budget']}
    - Interests : {user_requirements['interests']}
    - Group Type : {user_requirements['group_type']}
    - Nationality : {user_requirements['nationality']}

    ** Your Task : **
    Analyze the trip requirements and must use all the seven tools below one-by-one to create a comprehensive travel plan. You have access to below 7 specialized agents : 

    1. Weather and Season Information
    2. Transportation Planning 
    3. Accomodation recommendation
    4. Activities and attractions
    5. Budget optimization
    6. Saftey and local tips
    7. Travel documents and Visa requirements

    Use all the above 7 tools one by one strictly and fetch the information.
    # Focus on providing the most valuable information based on destination, duration and travel preferences. 

    # After consulting the necessary tools, provide a clear summary of your recommendations.
     After consulting the above 7 tools, provide a clear summary of your recommendations.

    """
    results = master_agent.invoke({"input": planning_prompt})

    complete_info = f"""

    Trip Requirements : 
    - Destination : {user_requirements['destination']}
    - Duration : {user_requirements['days']}
    - Travel Month : {user_requirements['travel_month']}
    - Budget Level : {user_requirements['budget']}
    - Interests : {user_requirements['interests']}
    - Group Type : {user_requirements['group_type']}
    - Nationality : {user_requirements['nationality']}

    Weather & Season Information : 
    {agent_results.get('weather', 'Not consulted by master agent')}

    Transport Plan : 
    {agent_results.get('transport', 'Not consulted by master agent')}

    Accomodations Recommendations : 
    {agent_results.get('accomodation', 'Not consulted by master agent')}

    Activities & Attractions: 
    {agent_results.get('activities', 'Not consulted by master agent')}

    Budget Breakdown : 
    {agent_results.get('budget', 'Not consulted by master agent')}

    Safety and Local Tips : 
    {agent_results.get('safety', 'Not consulted by master agent')}

    Travel Documents & Visa : 
    {agent_results.get('documents', 'Not consulted by master agent')}

    Master Agent Summary : 
    {results.get('output', 'No summary from Master agent available')}

    """

    agent_results['final_itinerary'] = generate_final_itinerary(complete_info)

    return agent_results


def get_agent_results():
    return agent_results

#
# user_requirements = {
#     'destination': input("Enter the destination : "),
#     'days': int(input('Enter the no of days : ')),
#     'budget': input("Budget level (low/medium/high)"),
#     'interests': input("Interests : (nature / adventure / food / meuseums : )"),
#     'travel_month': input('Travel Month : '),
#     'group_type': input('Group type (solo/couple/family) : '),
#     'nationality': input("You nationality : ") or "Indian"
# }
#
# results = run_travel_planning(user_requirements)
#
# for agent_name, result in results.items():
#     print(f'{agent_name} : \n')
#     print(result)
#     print('\n\n\n --------------------------------------')
#

#   to  Everyone
# git status
# git add .
# git commit -m "Deepa"
# git push