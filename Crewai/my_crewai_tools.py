'''
CREW AI :

Support Issues : Support Tickets (Service Now)
Problem Statement : How to efficiently analzye and provide resolution to these tickets ?


4 Agents :

1. Support Ticket Analyzer Agent : Understand the user's problem. Identify the root cause & assign the priority level.
2. Solution Specialist Agent : Technical Support Agent - Give the colution step by step
3. Reasearch & Verifier Agent : Verify the solution using a web search
4. Customer Response Writer : Email based on the solution
'''


import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from crewai import Agent , Task , Crew
from crewai_tools import SerperDevTool

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPEN_API_KEY')
os.environ['SERPER_API_KEY'] = '238a48fe4018074fc989652d0d73a8e11c6345d8'

llm = ChatOpenAI(model='gpt-4o-mini',temperature=0.2)

web_search_tool = SerperDevTool()

# Agent 1 :

analyzer = Agent(
    role = 'Support Ticket Analyzer',
    goal = 'Analyze customer support tickets to understand the issue and urgency',
    backstory = """ You  are an experienced customer support analyst who can quickly understand customer problems, identify root causes, and assess the priority level.
    You have handled thousands of support tickets across various categories.
    """,
    verbose = True,
    allow_delegation = False,
    llm = llm
)

# Agent 2 :

solution_specialist = Agent(
    role = 'Solution Specialist',
    goal = 'Provide detailed solutions and troubleshooting steps for customer issues',
    backstory = """ You are a technical support expert with deep knowledge across multiple domains.
    You excel at creating clear, step-by-step solutions that customers can easily follow.
    You always consider both technical and non-technical users
    """,
    verbose = True,
    allow_delegation = False,
    llm = llm
)

# Agent 3 :

research_agent = Agent(
    role = 'Research & Verification Specialist',
    goal = 'Research and verify solutions using real time web search to ensure accuracy',
    backstory = """ You are a diligent research specialist who verifies information before it reaches customers.
    You use web search to find latest documentation, known issues, official fixes , and community solutions.
    YOu cross-reference multiple sources to ensure the solution is accurate. 
    You specialize in finding real-world examples and verified troubleshooting steps.
    """,
    verbose = True,
    allow_delegation = False,
    llm = llm,
    tools = [web_search_tool]
)

# Agent 4 :
response_writer = Agent(
    role = 'Customer Response Writer',
    goal = 'Craft professional and empathetic customer response',
    backstory = """ You are a skilled customer service representative who specializes in writing clear. friendly , and professional responses. 
    You know how to balance technical accuracy with empathy and ensure customers feel heard and valued.
    """,
    verbose = True,
    allow_delegation = False,
    llm = llm
)






# Task 1 : Analyze the ticket

analyze_task = Task(

    description= """ Analyze the following customer support ticket : 

    {ticket}

    Provide : 

    - Issue category (Technical, Billing, Account, Product , Other)
    - Priority Level (Low, Medium, High, Critical)
    - Key Problem SUmmary
    - Customer Sentiment (Frustrated, Neutral, Satisfied, Confused)
    - Any relevent context or details
    """,
    agent= analyzer,
    expected_output = "A comprehensive analysis of the support ticket with categorization and priority assessment"
)


# Task 2 : Create a solution

solution_task = Task(

    description= """ Based on the ticket analysis  , create a detailed solution : 

    - Identify the root cause
    - Provide step-by-step troubleshooting or resolution steps
    - Include any relevant workarounds
    - Suggest preventive measures for the future
    - Add any technical details that support team needs to know

    """,
    agent=solution_specialist,
    expected_output= "A detailed solution with clear steps to resolve the customer's issue"

)

# Task 3 : Research and Verfiy

research_task = Task(

    description= """Research and verify the proposed solution using web search: 

    Use the web search tool for : 

    - Search for official documentation related to the issue
    - Find similar reported issues and their verified solutions
    - Check for any recent updates or patches
    - Verify the accuracy of the proposed solution
    - Find additional tips or alternative solutions 

    Provide :
    - Verification Status : (Verified/Needs Update/Alternative Found)
    - Links to official documentation (if found)
    - Any additional insights from web search
    - Confirmation that the solution is current and accurate

    """,
    agent=research_agent,
    expected_output= "A research report verifying the solution with sources and additional insights from web search"

)


# Task 4 : Write response


response_task = Task(

    description= """ Write a professional customer response email that : 

    - Acknowledge the customer's issue with empathy. 
    - Clearly explain the solution  in customer-friendly language
    - Provide step-by-step instructions
    - Offers additional help if needed
    - Maintain a professional yet friendly tone
    - Includes appropriate closing remarks

    The response should be ready to send directly to the customer
    """,
    agent=response_writer,
    expected_output= "A polished, professional customer response email ready to send"

)


support_crew = Crew(
    agents= [analyzer,solution_specialist,research_agent,response_writer],
    tasks=[analyze_task,solution_task,research_task,response_task],
    verbose=True
)

user_ticket = input('Ticket : ')

result = support_crew.kickoff(inputs={'ticket':user_ticket})

print('\n\n Result : \n\n')

print(result)

with open('crew_agent.txt','w',encoding='utf-8') as f:
    f.write(str(result))