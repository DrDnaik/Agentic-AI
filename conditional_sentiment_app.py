'''

--> A user input :

1. Analyze the sentiment : Positive , Negative, Neutral

2. If its a +ve sentiment : Positive Response

3. If it has a negative sentiment : Supportive Response

4. If it's a neutral : Neutral Response

'''

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


# Step 1 : Define our states


class GraphState(TypedDict):
    user_input: str
    sentiment: str
    response_type: str
    final_response: str


def analyze_sentiment(state: GraphState) -> GraphState:
    print('Analyzing Sentiment : ')
    user_input = state['user_input']
    prompt = f"""
    Analyze the sentiment of this text and respond with just one word - either 'positive' , 'negative' or neutral : {user_input}
    """
    response = llm.invoke(prompt)
    sentiment = response.content.strip().lower()
    state['sentiment'] = sentiment
    return state


def decide_response_type(state: GraphState) -> str:
    print('Deciding Response Type : ')

    sentiment = state['sentiment']
    if sentiment == 'positive':
        decision = "generate_positive_response"
    elif sentiment == "negative":
        decision = "generate_supportive_response"
    else:
        decision = "generate_neutral_response"

    return decision


def generate_positive_response(state: GraphState) -> GraphState:
    print('Generating Positive Response : ')
    user_input = state["user_input"]
    prompt = f"Generate an enthusiastic and positive response to : {user_input}"
    response = llm.invoke(prompt)
    state["response_type"] = "Happy Response"
    state["final_response"] = response.content
    return state


def generate_neutral_response(state: GraphState) -> GraphState:
    print('Generating Neutral Response : ')
    user_input = state["user_input"]
    prompt = f"Generate a balanced, informative response to : {user_input}"
    response = llm.invoke(prompt)
    state["response_type"] = "Neutral Response"
    state["final_response"] = response.content
    return state


def generate_supportive_response(state: GraphState) -> GraphState:
    print('Generating Supportive Response : ')
    user_input = state["user_input"]
    prompt = f"Generate a supportive and empathetic response to : {user_input}"
    response = llm.invoke(prompt)
    state["response_type"] = "Supportive Response"
    state["final_response"] = response.content
    return state


def build_conditional_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("decide_response_type", decide_response_type)
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


app = build_conditional_graph()
user_input = input('Write your thought : ')
initial_state = {
    "user_input": user_input,
    "sentiment": '',
    "response_type": '',
    "final_response": ''
}
result = app.invoke(initial_state)

print('Results : ')

print('Sentiment : ', result['sentiment'])
print('\n\nResponse Type : ', result['response_type'])
print('\n\nFinal Response : ', result['final_response'])


