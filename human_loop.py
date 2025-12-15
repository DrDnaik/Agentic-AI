"""
Human-in-the-Loop AI Feedback System

This system creates an iterative workflow where:
1. AI generates an initial answer to a user query
2. User provides feedback for improvements
3. AI refines the answer based on feedback
4. Process continues until user is satisfied or max iterations reached
"""

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Literal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPEN_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OPEN_API_KEY not found in environment variables")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize LLM
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)


class GraphState(TypedDict):
    """State schema for the workflow graph."""
    user_query: str
    current_answer: str
    feedback_history: List[str]
    iteration_count: int
    max_iterations: int
    user_satisfied: bool


def generate_initial_answer(state: GraphState) -> GraphState:
    """
    Generate the initial AI response to the user's query.

    Args:
        state: Current graph state containing user query

    Returns:
        Updated state with initial answer
    """
    user_query = state['user_query']

    prompt = f"""You are an AI Expert assistant. Generate a comprehensive and helpful answer to the following user query:

Query: {user_query}

Provide a clear, accurate, and well-structured response."""

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
        state['current_answer'] = answer
        state['iteration_count'] = 1

        logger.info("Generated initial answer")
        print(f"\n{'=' * 60}")
        print("INITIAL ANSWER:")
        print(f"{'=' * 60}")
        print(answer)
        print(f"{'=' * 60}\n")

    except Exception as e:
        logger.error(f"Error generating initial answer: {e}")
        state['current_answer'] = "Error generating answer. Please try again."

    return state


def improve_answer(state: GraphState) -> GraphState:
    """
    Improve the current answer based on user feedback.

    Args:
        state: Current graph state with feedback history

    Returns:
        Updated state with improved answer
    """
    current_answer = state['current_answer']
    user_query = state['user_query']
    latest_feedback = state['feedback_history'][-1]

    # Build feedback context
    feedback_context = "\n".join([
        f"- {fb}" for fb in state['feedback_history']
    ])

    prompt = f"""You previously provided this answer to the question: "{user_query}"

Previous Answer:
{current_answer}

User Feedback History:
{feedback_context}

Latest Feedback:
{latest_feedback}

Please improve your answer by:
1. Keeping the parts that are working well
2. Addressing the user's specific concerns from the feedback
3. Making the answer more clear, accurate, or comprehensive as needed

Provide the improved answer:"""

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
        state['current_answer'] = answer
        state['iteration_count'] += 1

        logger.info(f"Generated improved answer (iteration {state['iteration_count']})")
        print(f"\n{'=' * 60}")
        print(f"IMPROVED ANSWER (Iteration {state['iteration_count']}):")
        print(f"{'=' * 60}")
        print(answer)
        print(f"{'=' * 60}\n")

    except Exception as e:
        logger.error(f"Error improving answer: {e}")

    return state


def check_user_satisfied(state: GraphState) -> Literal["end", "continue"]:
    """
    Determine if the workflow should continue or end.

    Args:
        state: Current graph state

    Returns:
        "end" if satisfied or max iterations reached, "continue" otherwise
    """
    user_satisfied = state['user_satisfied']
    iteration_count = state['iteration_count']
    max_iterations = state['max_iterations']

    if user_satisfied:
        logger.info("User is satisfied with the answer")
        print("\n✓ User confirmed satisfaction!\n")
        return "end"
    elif iteration_count >= max_iterations:
        logger.warning("Maximum iterations reached")
        print(f"\n⚠ Maximum iterations ({max_iterations}) reached!\n")
        return "end"
    else:
        logger.info(f"Continuing feedback loop (iteration {iteration_count}/{max_iterations})")
        return "continue"


def get_user_feedback(state: GraphState) -> GraphState:
    """
    Collect user feedback on the current answer.

    Args:
        state: Current graph state

    Returns:
        Updated state with user feedback
    """
    print("\n" + "=" * 60)
    print("FEEDBACK REQUEST")
    print("=" * 60)

    try:
        feedback = input("\nYour feedback (or type 'satisfied'/'happy'/'ok' if you're satisfied): ").strip()

        if not feedback:
            feedback = "No feedback provided"

        # Check satisfaction keywords
        satisfied_keywords = ['satisfied', 'happy', 'ok', 'good', 'done', 'perfect']
        satisfied = any(keyword in feedback.lower() for keyword in satisfied_keywords)

        state['user_satisfied'] = satisfied
        state["feedback_history"].append(feedback)

        if satisfied:
            logger.info("User indicated satisfaction")
            print("✓ Understood - you're satisfied with the answer!")
        else:
            logger.info("User provided improvement feedback")
            print("✓ Feedback received - generating improved answer...")

    except KeyboardInterrupt:
        logger.info("User interrupted feedback collection")
        state['user_satisfied'] = True
        state["feedback_history"].append("User interrupted")
        print("\n\n⚠ Interrupted - finalizing with current answer...")
    except Exception as e:
        logger.error(f"Error collecting feedback: {e}")
        state['user_satisfied'] = True

    return state


def finalize_answer(state: GraphState) -> GraphState:
    """
    Display the final answer and summary.

    Args:
        state: Current graph state with final answer

    Returns:
        Unchanged state
    """
    current_answer = state['current_answer']
    iteration_count = state['iteration_count']

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(current_answer)
    print("=" * 60)
    print(f"\nTotal iterations: {iteration_count}")
    print(f"Feedback rounds: {len(state['feedback_history'])}")
    print("=" * 60 + "\n")

    logger.info("Workflow completed successfully")

    return state


def create_graph() -> StateGraph:
    """
    Create and configure the workflow graph.

    Returns:
        Compiled workflow application
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("generate_initial_answer", generate_initial_answer)
    workflow.add_node("improve_answer", improve_answer)
    workflow.add_node("get_user_feedback", get_user_feedback)
    workflow.add_node("finalize_answer", finalize_answer)

    # Set entry point
    workflow.set_entry_point("generate_initial_answer")

    # Define edges
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

    return workflow.compile()


def main():
    """
    Main entry point for the application.

    Returns:
        Final state after workflow completion
    """
    print("=" * 60)
    print("HUMAN-IN-THE-LOOP AI FEEDBACK SYSTEM")
    print("=" * 60)

    try:
        app = create_graph()

        user_query = input('\nWhat would you like to know? ').strip()

        if not user_query:
            print("No query provided. Exiting.")
            return None

        initial_state: GraphState = {
            "user_query": user_query,
            "current_answer": "",
            "feedback_history": [],
            "iteration_count": 0,
            "max_iterations": 5,
            "user_satisfied": False
        }

        logger.info(f"Starting workflow with query: {user_query}")
        result = app.invoke(initial_state)

        print("\n" + "=" * 60)
        print("CONVERSATION COMPLETED!")
        print("=" * 60 + "\n")

        return result

    except KeyboardInterrupt:
        print("\n\n⚠ Application interrupted by user.")
        logger.info("Application interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\n❌ An error occurred: {e}")
        return None


if __name__ == "__main__":
    result = main()