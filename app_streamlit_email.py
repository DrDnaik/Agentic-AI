'''

Email Answering Multi-Agent


Input : Emails from the customer

1. Email ANalyzer Agent ( Analyze the email - type of email (enquiry, compliant, compliment) , formal  / friendly )

2. Response Write Agent ( Write a response )

3. Polish email Agent
'''

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict
import streamlit as st

load_dotenv()  # env
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)


# Step 1 : States

class GraphState(TypedDict):
    email: str
    email_type: str
    response_tone: str
    generated_email: str
    final_email: str


def email_analyzer_agent(state: GraphState) -> GraphState:
    email = state['email']
    prompt = f"""
    You are an EMail ANalyzer Agent . Please analyze this email and determine 

    1. Email Type (complaint / inquiry / compliment / request etc.)
    2. Appropriate Response Tone : (formal , friendly, apologetic etc. )

    Email : {email}

    Respond in this format only : 

    Type : [email_type]
    Tone : [response_tone]
    """

    response = llm.invoke(prompt)
    analysis = response.content.strip()

    lines = analysis.split('\n')

    print(lines)

    for line in lines:
        if line.startswith("Type :"):
            email_type = line.split("Type :")[1].strip()
        elif line.startswith("Tone :"):
            tone = line.split("Tone :")[1].strip()

    # Update state
    state["email_type"] = email_type
    state["response_tone"] = tone

    print(f"Email Analyzer Agent completed - Type: {email_type}, Tone: {tone}")
    return state


def response_writer_agent(state: GraphState) -> GraphState:
    user_email = state['email']
    email_type = state["email_type"]
    response_tone = state["response_tone"]

    prompt = f"""
    You are a Responsive Email Writer Agent. You have to write a response depending upon the type and tone for the below email.

    Original EMail : {user_email}
    Tone of the email you have to give : {response_tone}
    Email Type : {email_type}

    As an expert Response Writer Agent, generate a {response_tone} response that appropriately addresses this {email_type}.
    Make it professional but {response_tone}. Include:
    - Appropriate greeting
    - Response to their concern/question
    - Next steps if needed
    - Professional closing

    Response : """

    response = llm.invoke(prompt)
    analysis = response.content.strip()
    state["generated_email"] = analysis
    print('Response Agent completed - Value is : ', analysis)
    return state


def polish_email_agent(state: GraphState) -> GraphState:
    old_email = state["generated_email"]
    email_type = state["email_type"]
    response_tone = state["response_tone"]

    prompt = f"""
    You are an Email Polish Agent. Polish this email response for the final delivery.

    Original Response : {old_email}

    As an expert Polish Agent, make sure it : 

    - Has proper email formatting
    - Is gramatically correct and perfect
    - Maintains the {response_tone} tone
    - Its appropriate for a {email_type}
    - Includes subject line suggestion

    Format the final email properly with subject Name

    Final Email:"""

    response = llm.invoke(prompt)
    final_email = response.content.strip()

    state['final_email'] = final_email
    print('Final EMail is : ')
    print('\n', final_email)
    return state


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("analyze_email", email_analyzer_agent)
    workflow.add_node("response_email", response_writer_agent)
    workflow.add_node("polish_email", polish_email_agent)

    workflow.set_entry_point("analyze_email")
    workflow.add_edge("analyze_email", "response_email")
    workflow.add_edge("response_email", "polish_email")
    workflow.add_edge("polish_email", END)

    app = workflow.compile()
    return app


def process_email(email):
    app = build_graph()

    initial_state = {
        "email": email,
        "email_type": '',
        "response_tone": ''
    }

    result = app.invoke(initial_state)
    return result


# Streamlit Interface :


def main():
    st.set_page_config(
        page_title="Walmart's Email Generator",
        page_icon="📧",
        layout="wide"
    )

    st.title("Walmart's Email Generator")
    st.markdown('##### Generate professional email responses using AI Powered Langgraph Agents')

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader('Input Email from Customer')

        user_email = st.text_area(
            "Enter the email you want",
            height=200,
            placeholder="Paste the email content here ..."
        )

    with col2:
        st.subheader('Agent Response')

        if st.button('Generate Response'):
            if user_email:
                with st.spinner('AI Agent email is getting generated ...'):
                    result = process_email(user_email)

                st.success('Email generated successfully')

                st.subheader('AI Generated email')
                st.text_area("Your AI Generated email",
                             value=result['final_email'],
                             height=200)

    st.markdown("---")
    st.markdown('Developed by Deepa')


main()

    # with col1:
    #     st.markdown('Input Email from Customer')
    #
    #     user_email = st.text_area(
    #         "Enter the email you want",
    #         height= 200,
    #         placeholder="Paste the email content here ..."
    #     )
    #
    #     if st.button('Generate Response'):
    #         st.markdown('Agent Response')
    #         result = process_email(user_email)
    #         st.text_area(
    #             "Email Response",
    #             value=result['generated_email'],
    #             height= 300
    #         )
    #



