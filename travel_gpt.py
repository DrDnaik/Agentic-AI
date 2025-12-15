import os
import requests
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY")

# --------------------------------------------
# LLM
# --------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

agent_results = {}  # global collector


# =========================================================
# 1. REAL WEATHER LOOKUP TOOL (non-LLM)
# =========================================================
def get_weather(city):
    """Fetches real-time weather from Open-Meteo."""
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    geo_data = requests.get(geo_url).json()

    lat = geo_data["results"][0]["latitude"]
    lon = geo_data["results"][0]["longitude"]

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current_weather=true"
    )

    weather = requests.get(weather_url).json()
    temp = weather["current_weather"]["temperature"]
    wind = weather["current_weather"]["windspeed"]

    return f"Current temperature in {city}: {temp}°C, wind {wind} km/hr"


# =========================================================
# 2. WEATHER & SEASON AGENT
# =========================================================
def weather_agent(destination, travel_month):
    current_month = datetime.now().strftime("%B")

    realtime = ""
    if travel_month.lower() == current_month.lower():
        realtime_data = get_weather(destination)
        realtime = f"\nReal-time weather:\n{realtime_data}\n"

    prompt = f"""
You are a weather and season advisor.

Destination: {destination}
Travel Month: {travel_month}
{realtime}

Provide:
1) Expected weather
2) Best time to visit
3) Packing tips
4) Seasonal festivals
5) Weather warnings
"""

    resp = llm.invoke(prompt).content
    agent_results["weather"] = resp
    return resp


# =========================================================
# 3. TRANSPORT AGENT
# =========================================================
def transport_agent(destination, days, budget):
    prompt = f"""
You are a transport route planner.

Destination: {destination}
Duration: {days} days
Budget: {budget}

Provide:
1) Best ways to reach (flight/train/bus)
2) Local transport
3) Cost comparison
4) Travel times
5) Booking tips
"""
    resp = llm.invoke(prompt).content
    agent_results["transport"] = resp
    return resp


# =========================================================
# 4. ACCOMMODATION AGENT
# =========================================================
def accommodation_agent(destination, budget, group_type, days):
    prompt = f"""
You are an accommodation specialist.

Destination: {destination}
Budget: {budget}
Group Type: {group_type}
Duration: {days}

Provide:
1) Types of accommodations
2) Best areas to stay
3) Price ranges
4) Amenities
5) Booking platforms
6) 3 hotel recommendations with pros/cons
"""
    resp = llm.invoke(prompt).content
    agent_results["accommodation"] = resp
    return resp


# =========================================================
# 5. ACTIVITIES AGENT
# =========================================================
def activities_agent(destination, interests, days, budget):
    prompt = f"""
You are an activity & attractions planner.

Destination: {destination}
Interests: {interests}
Duration: {days}
Budget: {budget}

Create a daily plan with:
1) Attractions with timings
2) Food experiences
3) Free activities
4) Cultural experiences
5) Adventure options
6) Shopping areas
7) Nightlife options
"""

    resp = llm.invoke(prompt).content
    agent_results["activities"] = resp
    return resp


# =========================================================
# 6. BUDGET AGENT
# =========================================================
def budget_agent(destination, days, budget, group_type):
    prompt = f"""
You are a budget optimizer.

Destination: {destination}
Duration: {days}
Budget Level: {budget}
Group Type: {group_type}

Provide:
- Transportation cost
- Accommodation estimate
- Food cost
- Activities
- Shopping & misc
- Emergency buffer
- Total cost (budget/mid/comfortable)
"""
    resp = llm.invoke(prompt).content
    agent_results["budget"] = resp
    return resp


# =========================================================
# 7. SAFETY AGENT
# =========================================================
def safety_agent(destination, group_type):
    prompt = f"""
You are a safety advisor.

Destination: {destination}
Group Type: {group_type}

Provide:
1) Safety tips
2) Areas to avoid
3) Transport safety
4) Emergency numbers
5) Etiquette
6) Common scams
7) Useful phrases
8) Tipping customs
"""
    resp = llm.invoke(prompt).content
    agent_results["safety"] = resp
    return resp


# =========================================================
# 8. DOCUMENTS AGENT
# =========================================================
def documents_agent(destination, nationality):
    prompt = f"""
You are a visa/document expert.

Destination: {destination}
Nationality: {nationality}

Provide:
1) Visa requirements
2) Passport validity rules
3) Required documents
4) Processing time & fees
5) Vaccination needs
6) Travel insurance suggestions
7) Customs rules
"""
    resp = llm.invoke(prompt).content
    agent_results["documents"] = resp
    return resp


# =========================================================
# 9. FINAL ITINERARY GENERATOR
# =========================================================
def generate_final_itinerary(all_info):
    prompt = f"""
You are an itinerary generator.

Use the following collected info:
{all_info}

Create a complete itinerary with:
- Trip overview
- Day-by-day plan
- Transportation
- Meals
- Pre-trip checklist
- Budget summary
- Packing list
- Important contacts
"""
    return llm.invoke(prompt).content


# =========================================================
# MASTER ORCHESTRATOR → predictable execution
# =========================================================
def run_travel_planning(req):
    global agent_results
    agent_results = {}  # reset

    # Sequential deterministic tool execution
    weather_agent(req["destination"], req["travel_month"])
    transport_agent(req["destination"], req["days"], req["budget"])
    accommodation_agent(req["destination"], req["budget"], req["group_type"], req["days"])
    activities_agent(req["destination"], req["interests"], req["days"], req["budget"])
    budget_agent(req["destination"], req["days"], req["budget"], req["group_type"])
    safety_agent(req["destination"], req["group_type"])
    documents_agent(req["destination"], req["nationality"])

    # Combine everything
    all_info = "\n\n".join(
        f"{k.upper()}:\n{v}" for k, v in agent_results.items()
    )

    agent_results["final_itinerary"] = generate_final_itinerary(all_info)
    return agent_results


if __name__ == "__main__":
    user_requirements = {
        "destination": input("Enter the destination: "),
        "days": int(input("Enter number of days: ")),
        "budget": input("Budget (low/medium/high): "),
        "interests": input("Interests: "),
        "travel_month": input("Travel month: "),
        "group_type": input("Group type (solo/couple/family): "),
        "nationality": input("Nationality: ") or "Indian"
    }

    results = run_travel_planning(user_requirements)

    for agent, output in results.items():
        print(f"\n=== {agent.upper()} ===\n")
        print(output)
