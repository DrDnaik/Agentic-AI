'''

--> Human asks a query

--> AI will give a response

   --> Human have to validate it
   --> If there is a improvement required, he has to tell that. (Iterations)
   --> If user is satisfied, make it as a final answer

'''

'''
Stock Alerts --> AI --> [ R1,R2...] (DB)  --> Approve --> Output 
Competitors --> AI (Structure)  --> DB1 --> (Human approve) --> [DB2]
'''

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

load_dotenv()  # env
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)


class GraphState(TypedDict):
    user_query: str
    current_answer: str
    feedback_history: List[str]
    iteration_count: int
    max_iterations: int
    user_satisfied: bool


def generate_initial_answer(state: GraphState) -> GraphState:
    user_query = state['user_query']

    prompt = f""" You are a AI Expert. Generate a detailed answer to the user query : {user_query}
    Provide a clear, accurate, and well-structured response.
    """

    response = llm.invoke(prompt)
    answer = response.content.strip()
    state['current_answer'] = answer
    state['iteration_count'] = 1

    print('Initial Answer : ', answer)

    return state


def improve_answer(state: GraphState) -> GraphState:
    current_answer = state['current_answer']
    user_query = state['user_query']
    latest_feedback = state['feedback_history'][-1]

    prompt = f"""You  previously provided this answer to the question {user_query}" : 
    {current_answer}

    The user has provided the feedback : 
    {latest_feedback}

    Please improve your answer by:
1. Keeping the parts that are working well
2. Addressing the user's specific concerns from the feedback
3. Making the answer more clear, accurate, or comprehensive as needed
    """

    response = llm.invoke(prompt)
    answer = response.content.strip()
    state['current_answer'] = answer
    state['iteration_count'] += 1

    print('Improved Answer : ', answer)

    return state


def check_user_satisfied(state: GraphState) -> str:
    user_satisfied = state['user_satisfied']
    iteration_count = state['iteration_count']
    max_iterations = state['max_iterations']

    if user_satisfied:
        # print('User is satisfied finally!!')
        return "end"

    elif iteration_count >= max_iterations:
        print('Max iterations reached!!')
        return "end"
    else:
        print('Continuing to get more feedback ...')
        return "continue"


def get_user_feedback(state: GraphState) -> GraphState:
    print('Getting the user feedback ...')

    feedback = input("Your feedback :  (or if satisfied , reply with 'satisfied' or 'happy')")
    satisfied = feedback.lower() in ['satisfied', 'happy', 'ok', 'good', 'perfect', 'thanks', 'thank you']
    state['user_satisfied'] = satisfied
    state["feedback_history"].append(feedback)

    if satisfied:
        print('User is satisfied')
    else:
        print('User is not satisfied')

    return state


def finalize_answer(state: GraphState) -> GraphState:
    current_answer = state['current_answer']
    print('Final Answer : ')
    print('\n', current_answer)
    return state


def create_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("generate_initial_answer", generate_initial_answer)
    workflow.add_node("improve_answer", improve_answer)
    workflow.add_node("check_user_satisfied", check_user_satisfied)
    workflow.add_node("finalize_answer", finalize_answer)
    workflow.add_node("get_user_feedback", get_user_feedback)

    workflow.set_entry_point("generate_initial_answer")

    workflow.add_edge("generate_initial_answer", "get_user_feedback")

    workflow.add_conditional_edges(
        "get_user_feedback",
        check_user_satisfied,
        {
            "continue": "improve_answer",
            "end": "finalize_answer"
        }
    )

    workflow.add_edge("improve_answer", "get_user_feedback")
    workflow.add_edge("finalize_answer", END)

    app = workflow.compile()
    return app


def main():
    app = create_graph()

    user_query = input('What you would like to know ? ')
    print("printing user query",user_query)
    if user_query== "" or user_query == None:
        print("Exiting form the app as no user query")
        return None
    initial_state = {
        "user_query": user_query,
        "current_answer": "",
        "feedback_history": [],
        "iteration_count": 0,
        "max_iterations": 5,
        "user_satisfied": False
    }

    result = app.invoke(initial_state)

    print('CONVERSATION COMPLETED!!!')

    return result


result = main()
