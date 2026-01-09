import os
from dotenv import load_dotenv
import requests
import json
from agents import Agent, function_tool, Runner

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPEN_API_KEY')
# os.environ['SERPER_API_KEY'] = '238a48fe4018074fc989652d0d73a8e11c6345d8'

'''
3 Agents : 

1. Market Research Agent : Uses serper tool to search for stock prices, news, market data. 
2. Analysis Agent : Analyzes financial data and market sentiment
3. Investment Advisor Agent : Genrate personalized investment recommendations

'''
SERPER_API_KEY = '238a48fe4018074fc989652d0d73a8e11c6345d8'


@function_tool
def serper_search(query: str) -> str:
    """
    Perform a Google search using Serper API
    """
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"q": query}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return json.dumps(response.json())
    except Exception as e:
        return {"error": str(e)}


# Agent 1 : Market Research Agent

market_research_agent = Agent(
    name='Market Research Agent',
    model='gpt-4o-mini',
    tools=[serper_search],
    instructions="""
    YOu are a financial market research specialist. 

    Steps : 
    1. Perform multiple serper searches for the given stock. 
    2. Extract : 
       - Stock price info
       - Recent news
       - Analyst opinions
       - Market sentiment
    3. Synthesize findings into a structured research report.

    Be factual and objective.
    """
)

# Agent 2 : Analysis Agent

analysis_agent = Agent(
    name=' Analysis Agent',
    model='gpt-4o-mini',
    instructions="""
    YOu are a senior financial analyst. 

    Analyze the provided market research and evaulate :

    - Valuation
    - Technical Trends
    - Fundamental strength
    - Risks
    - Opportunity score (1-10)
    - Time horizon recommendations

    Be analytical and balanced.
    """
)

# Agent 3 : Investment Advisor Agent

investor_advisor_agent = Agent(
    name='Investment Advisor Agent',
    model='gpt-4o-mini',
    instructions="""
    You are a certified investment advisor. 

    Create a personalized investment recommendation including : 

    - BUY / HOLD / SELL
    - Entry Range
    - Stop Loss 
    - Target price
    - Portfolio Allocation
    - Monitoring Plan
    - Risk Disclaimer

    Be professional and client-focused. 
    """
)


def run_stock_analysis(stock_symbol, company_name, investment_amount, risk_profile):
    researcher_prompt = f"""
    Research stock : {company_name} ({stock_symbol})

    Run searches: 
    - "{stock_symbol} stock price today"
    - "{company_name} latest news"
    - "{stock_symbol} analyst rating"
    - "{company_name} earnings performance"
    - "{stock_symbol} investor sentiment"

    Return a structured market research report
    """

    research_result = Runner.run_sync(market_research_agent,
                                      input=researcher_prompt)

    research_summary = research_result.final_output

    print('\n\nAgent 1 : Research Summary - ', research_summary)

    analysis_prompt = f"""
    Stock : {company_name} ({stock_symbol})

    Investment amount : ${investment_amount}
    Risk profile : {risk_profile}

    Market Research : {research_summary}

    Provide a deep financial analysis.
    """

    analysis_result = Runner.run_sync(analysis_agent,
                                      input=analysis_prompt)

    analysis_summary = analysis_result.final_output

    print('\n\nAgent 2 : Analysis Summary - ', analysis_summary)

    advisor_prompt = f"""

    Client Profile : 
    - Stock : {company_name} ({stock_symbol})
    - Investment amount : ${investment_amount}
    - Risk profile : {risk_profile}

    Market Research : {research_summary}

    Financial Analysis : {analysis_summary}

    Generate a final investment recommendation.
    """

    advisor_result = Runner.run_sync(investor_advisor_agent,
                                     input=advisor_prompt)

    advisor_summary = advisor_result.final_output

    print('\n\nAgent 3 : Final AI Stock Recommendation :  \n\n ', advisor_summary)

    return advisor_summary


stock_symbol = input('Stock Symbol (Eg : AAPL , MSFT , TSLA) : ').strip().upper()
company_name = input('Company Name : ')
investment_amount = int(input('Amount you are interested to invest :'))
risk_profile = input('Risk Tolerance (Conservative / Moderate / Aggressive ) : ')

advisor_summary = run_stock_analysis(stock_symbol, company_name, investment_amount, risk_profile)