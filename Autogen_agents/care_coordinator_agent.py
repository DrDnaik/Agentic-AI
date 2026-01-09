'''
Medical Diagnosis Assistant :

Patient (User inputs)

1. General Practitioner --> COllecting all the info from the patient. (info collecting)
2. Specialist : GP - Analyze and give the treament
3. Medical Advisor : Patiently firendly , lifestyle advice

'''

import autogen
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPEN_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


config_list = [
    {
        "model":"gpt-4o-mini",
        "api_key":OPENAI_API_KEY
    }
]

llm_config = {
    "config_list":config_list,
    "temperature":0.3
}


def create_agents():

    # AGent 1 : Patient Agent

    patient = autogen.UserProxyAgent(
        name="Patient",
        system_message=(
            "You are a patient"
            "Answer the doctor's question honestly and concisely."
        ),
        human_input_mode="ALWAYS",
        code_execution_config=False
    )

    # Agent 2 : General Practitioner

    gp = autogen.AssistantAgent(
        name = "General_Practitioner",
        system_message="""
    You are a General Practitioner(GP).

    Responsibilities: 

    - Ask all necessary follow-up questions to the Patient.
    - Collect missing information from the patient.
    - Do not provide a final diagnosis. 
    - Once sufficient information is collected, say : 
    "HANDOVER TO SPECIALIST : [structured summary of symptoms, duration, severity, relevant history]"
    - Then type TERMINATE to end your role.
    """,
    llm_config=llm_config
    )

    # Agent 3 : Specialist


    specialist = autogen.AssistantAgent(
        name = "Specialist",
        system_message="""
    You are a Medical Specialist.

    Rules: 

    - Wait for GP to provide "HANDOVER TO SPECIALIST" message.
    - Do not ask the patient any questions.
    - ONLY analyze the GP's summary.
    - Provide : 
        1. Diagnosis
        2. Recommended Tests
        3. Initial treatment suggestions
    - End with "HANDOVER TO ADVISOR" and type TERMINATE

    """,
    llm_config=llm_config
    )

    # AGent 4 : Medical Advisor


    advisor = autogen.AssistantAgent(
        name = "Medical_Advisor",
        system_message="""
    You are a Medical Advisor.

    Responsibilities: 

    - Wait for "HANDOVER TO ADVISOR" message.
    - Review the specialist's assessment.
    - Provide : 
        - Lifestyle advice.
        - Preventive measures
        - Follow-up guidance
    - Summarize in simple language.
    - ALWAYS end with a medical disclaimer.
    - Type TERMINATE when done.

    """,
    llm_config=llm_config
    )

    return patient , gp, specialist , advisor


def run_medical_consultation():

    initial_symptoms = input("How are you feeling today? What are your symptoms? ").strip()

    patient , gp, specialist , advisor = create_agents()

    # Creating a group chat with all the agents

    groupchat = autogen.GroupChat(
        agents=[patient , gp, specialist , advisor],
        messages=[],
        max_round=15,
        speaker_selection_method="auto"
    )

    # Create a group chat manager to manage our group chat

    manager = autogen.GroupChatManager(groupchat=groupchat,llm_config=llm_config)

    patient.initiate_chat(
        manager,
        message=f"""
    I am experiencing the following symptoms :
    {initial_symptoms}

    GP : Please collect necessary information from me, then summarize.
    Specialist : After GP summaary, provide your diagnosis.
    Advisior : After specialist, provide final advice.

        """
    )

if __name__ == "__main__":
    run_medical_consultation()