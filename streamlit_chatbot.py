'''
Streamlit Chatbot with Sentiment-Based Responses

A user input:
1. Analyze the sentiment: Positive, Negative, Neutral
2. If it's a +ve sentiment: Positive Response
3. If it has a negative sentiment: Supportive Response
4. If it's a neutral: Neutral Response
'''

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict

load_dotenv()  # env
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)


# Step 1: Define our states

class GraphState(TypedDict):
    user_input: str
    sentiment: str
    response_type: str
    final_response: str


def analyze_sentiment(state: GraphState) -> GraphState:
    user_input = state['user_input']
    prompt = f"""
    Analyze the sentiment of this text and respond with just one word - either 'positive', 'negative' or 'neutral': {user_input}
    """
    response = llm.invoke(prompt)
    sentiment = response.content.strip().lower()
    state['sentiment'] = sentiment
    return state


def decide_response_type(state: GraphState) -> str:
    sentiment = state['sentiment']
    if sentiment == 'positive':
        decision = "generate_positive_response"
    elif sentiment == "negative":
        decision = "generate_supportive_response"
    else:
        decision = "generate_neutral_response"

    return decision


def generate_positive_response(state: GraphState) -> GraphState:
    user_input = state["user_input"]
    prompt = f"Generate an enthusiastic and positive response to: {user_input}"
    response = llm.invoke(prompt)
    state["response_type"] = "Happy Response"
    state["final_response"] = response.content
    return state


def generate_neutral_response(state: GraphState) -> GraphState:
    user_input = state["user_input"]
    prompt = f"Generate a balanced, informative response to: {user_input}"
    response = llm.invoke(prompt)
    state["response_type"] = "Neutral Response"
    state["final_response"] = response.content
    return state


def generate_supportive_response(state: GraphState) -> GraphState:
    user_input = state["user_input"]
    prompt = f"Generate a supportive and empathetic response to: {user_input}"
    response = llm.invoke(prompt)
    state["response_type"] = "Supportive Response"
    state["final_response"] = response.content
    return state


def build_conditional_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("generate_positive_response", generate_positive_response)
    workflow.add_node("generate_neutral_response", generate_neutral_response)
    workflow.add_node("generate_supportive_response", generate_supportive_response)

    workflow.set_entry_point("analyze_sentiment")
    workflow.add_conditional_edges("analyze_sentiment",
                                   decide_response_type,
                                   {
                                       "generate_positive_response": "generate_positive_response",
                                       "generate_neutral_response": "generate_neutral_response",
                                       "generate_supportive_response": "generate_supportive_response"
                                   })

    workflow.add_edge("generate_positive_response", END)
    workflow.add_edge("generate_neutral_response", END)
    workflow.add_edge("generate_supportive_response", END)

    app = workflow.compile()
    return app


def process_user_input(user_input):
    """Process user input through the conditional graph"""
    app = build_conditional_graph()
    initial_state = {
        "user_input": user_input,
        "sentiment": '',
        "response_type": '',
        "final_response": ''
    }
    result = app.invoke(initial_state)
    return result


def main():
    st.set_page_config(
        page_title='AI Agent Sentiment BOT',
        page_icon='🤖',
        layout='wide'
    )

    st.title('AI Agent Sentiment BOT')
    st.markdown('##### Chat with AI that responds based on your sentiment')

    if "messages" not in st.session_state:
        st.session_state.messages = []

        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I am your AI chatbot. I analyze your sentiment and respond accordingly. How are you feeling today ?",
            "sentiment": "neutral",
            "response_type": "welcome message"
        })

    with st.sidebar:

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Chat Cleared! How can I help you today?",
                "sentiment": "neutral",
                "response_type": "System message"
            }]

    chat_container = st.container()

    # is for displaying the chat messages :

    with chat_container:
        for message in st.session_state.messages:
            if message['role'] == 'user':
                with st.chat_message("user", avatar="👤"):
                    st.write(message['content'])

            if message['role'] == 'assistant':
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(message['content'])

    # CHat input

    if user_input := st.chat_input("Type your message here ..."):
        st.session_state.messages.append({'role': 'user', "content": user_input})

        with st.chat_message("user", avatar="👤"):
            st.write(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking ..."):
                result = process_user_input(user_input)

                # Display AI response
                st.write(result['final_response'])

                col1, col2 = st.columns([1, 1])
                with col1:
                    sentiment_emoji = {
                        "positive": "😊",
                        "negative": "😢",
                        "neutral": "😐"
                    }
                    st.caption(f"Sentiment : {sentiment_emoji.get(result['sentiment'])} {result['sentiment']}")

                with col2:
                    st.caption(f"Response Type : {result['response_type']}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result['final_response']
        })


main()















