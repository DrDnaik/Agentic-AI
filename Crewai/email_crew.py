'''
CREW AI :


Problem Statement : User wants to write and respond to email . User will give  topic or a brief input on
what user wants to be written in the email.
How to efficiently analyze user input and provide well crafted email based on user requirements.


3 Agents :

1.User query Analyzer Agent : Understand the user's query
2. Email Specialist Agent : Write a well crafted email base don user query
3. Grammar and spell check : Check the generated email for spellings and grammar
'''


import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from crewai import Agent , Task , Crew

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPEN_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini',temperature=0.2)





# Agent 1: User Query Analyzer
query_analyzer = Agent(
    role='Email Request Analyzer',
    goal='Understand the user topic/context and extract key details for email writing',
    backstory="""You are an expert at understanding what users want to communicate via email.
    When a user provides a topic or brief context like "write to my son's teacher about homework" 
    or "email my colleague about project deadline", you quickly identify:
    - Who the recipient is (teacher, colleague, manager, friend, etc.)
    - What the main topic/purpose is
    - The appropriate tone needed (formal, casual, friendly, professional)
    - Key points that should be covered
    You acknowledge the request and provide clear guidance for the email writer.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Agent 2: Email Writer
email_writer = Agent(
    role='Professional Email Writer',
    goal='Write well-crafted emails based on the analyzed topic and context',
    backstory="""You are a skilled email writer who can create emails for any situation.
    Whether it's writing to a teacher about a student's progress, emailing a colleague 
    about work matters, or reaching out to a client, you know exactly how to structure 
    and phrase emails appropriately. You include proper greetings, clear body content, 
    and appropriate closings. You match the tone to the recipient and purpose.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Agent 3: Grammar and Spell Checker
grammar_checker = Agent(
    role='Grammar and Spell Check Specialist',
    goal='Check and correct grammar, spelling, and ensure the email is polished',
    backstory="""You are a meticulous editor who reviews every email for:
    - Spelling mistakes
    - Grammar errors
    - Punctuation issues
    - Sentence structure and clarity
    - Professional formatting
    You make sure the final email is error-free and ready to send.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)


# Task 1: Analyze user's topic/query
analyze_task = Task(
    description="""The user wants to write an email. Here is their topic/context:

    {user_input}

    Analyze and identify:
    1. Who is the recipient? (teacher, colleague, manager, client, etc.)
    2. What is the main purpose/topic of the email?
    3. What tone should be used? (formal, semi-formal, casual, friendly)
    4. What key points should be included in the email?
    5. Any special considerations?

    Acknowledge the request and provide a clear breakdown for the email writer.""",
    agent=query_analyzer,
    expected_output="A clear analysis acknowledging the user's request with recipient type, purpose, tone, and key points identified"
)

# Task 2: Write the email
write_task = Task(
    description="""Based on the analysis, write a complete email that includes:

    1. Subject Line (appropriate for the topic)
    2. Greeting (Dear [Name], Hi [Name], Hello [Name], etc.)
    3. Opening paragraph (introduce the purpose)
    4. Main content (cover all key points clearly)
    5. Closing paragraph (call to action or polite ending)
    6. Sign-off (Best regards, Sincerely, Thanks, etc.)

    Make sure the email is well-structured, clear, and matches the required tone.
    The email should be ready to send with just the recipient's name filled in.""",
    agent=email_writer,
    expected_output="A complete, well-written email with subject line, proper greeting, body, closing, and sign-off",
    context=[analyze_task]
)

# Task 3: Grammar and spell check
check_task = Task(
    description="""Review the written email and:

    1. Correct any spelling errors
    2. Fix grammar mistakes
    3. Check punctuation
    4. Improve sentence clarity if needed
    5. Ensure proper formatting and spacing
    6. Verify the tone is consistent
    7. Make sure it flows well and is professional

    Provide the final, polished version of the email ready to send.""",
    agent=grammar_checker,
    expected_output="A final, error-free, polished email ready to send to the recipient",
    context=[write_task]
)




email_crew = Crew(
    agents=[query_analyzer, email_writer, grammar_checker],
    tasks=[analyze_task, write_task, check_task],
    verbose=True
)

user_query= input('Query : ')

result = email_crew.kickoff(inputs={'user_input':user_query})

print('\n\n Result : \n\n')

print(result)

with open('crew_email_agent.txt','w',encoding='utf-8') as f:
    f.write(str(result))


