'''
Human-in-the-Loop AI System with Streamlit UI

--> Human asks a query
--> AI will give a response
--> Human validates and provides feedback
--> AI improves based on feedback (Iterations)
--> Process continues until user is satisfied
'''

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

load_dotenv()
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

    prompt = f"""You are an AI Expert. You generate an answer to the user query: {user_query}
    """

    response = llm.invoke(prompt)
    answer = response.content.strip()
    state['current_answer'] = answer
    state['iteration_count'] = 1

    return state


def improve_answer(state: GraphState) -> GraphState:
    current_answer = state['current_answer']
    user_query = state['user_query']
    latest_feedback = state['feedback_history'][-1]

    prompt = f"""You provided this answer to the question "{user_query}":
    {current_answer}

    The user has provided the feedback:
    {latest_feedback}

    Please improve your answer based on the feedback. Keep what's good and address the user's concerns.
    """

    response = llm.invoke(prompt)
    answer = response.content.strip()
    state['current_answer'] = answer
    state['iteration_count'] += 1

    return state


def check_user_satisfied(state: GraphState) -> str:
    user_satisfied = state['user_satisfied']
    iteration_count = state['iteration_count']
    max_iterations = state['max_iterations']

    if user_satisfied:
        return "end"
    elif iteration_count >= max_iterations:
        return "end"
    else:
        return "continue"


def create_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("generate_initial_answer", generate_initial_answer)
    workflow.add_node("improve_answer", improve_answer)

    workflow.set_entry_point("generate_initial_answer")
    workflow.add_edge("generate_initial_answer", END)
    workflow.add_edge("improve_answer", END)

    app = workflow.compile()
    return app


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Human-in-the-Loop AI",
        page_icon="🔄",
        layout="wide"
    )

    st.title("🔄 Human-in-the-Loop AI System")
    st.markdown("##### Iteratively improve AI responses with your feedback!")

    # Initialize session state
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False
    if 'current_state' not in st.session_state:
        st.session_state.current_state = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 How It Works")
        st.markdown("""
        1. **Ask** - Enter your question
        2. **Review** - AI provides an answer
        3. **Feedback** - Give improvement suggestions
        4. **Iterate** - AI refines the answer
        5. **Approve** - Mark as satisfied when done
        """)

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        max_iterations = st.slider("Max Iterations", 1, 10, 5)

        st.markdown("---")
        if st.button("🔄 Start New Conversation", use_container_width=True):
            st.session_state.conversation_started = False
            st.session_state.current_state = None
            st.session_state.conversation_history = []
            st.rerun()

    # Main content area
    if not st.session_state.conversation_started:
        # Initial query input
        st.markdown("### 💬 What would you like to know?")

        col1, col2 = st.columns([3, 1])
        with col1:
            user_query = st.text_area(
                "Enter your question:",
                height=100,
                placeholder="Ask me anything... e.g., 'Explain quantum computing', 'How to learn Python?'"
            )

        with col2:
            st.markdown("**💡 Example Questions:**")
            st.markdown("- Explain machine learning")
            st.markdown("- How to stay productive?")
            st.markdown("- What is blockchain?")

        if st.button("🚀 Generate Answer", type="primary", use_container_width=True):
            if user_query.strip():
                with st.spinner("🤖 AI is thinking..."):
                    # Create initial state
                    app = create_graph()
                    initial_state = {
                        "user_query": user_query,
                        "current_answer": "",
                        "feedback_history": [],
                        "iteration_count": 0,
                        "max_iterations": max_iterations,
                        "user_satisfied": False
                    }

                    # Generate initial answer
                    result = app.invoke(initial_state)

                    # Update session state
                    st.session_state.current_state = result
                    st.session_state.conversation_started = True
                    st.session_state.conversation_history.append({
                        "type": "query",
                        "content": user_query
                    })
                    st.session_state.conversation_history.append({
                        "type": "answer",
                        "content": result['current_answer'],
                        "iteration": result['iteration_count']
                    })

                    st.success("✅ Initial answer generated!")
                    st.rerun()
            else:
                st.warning("⚠️ Please enter a question!")

    else:
        # Show conversation history
        st.markdown("### 📜 Conversation History")

        # Display the original query
        query = next((item['content'] for item in st.session_state.conversation_history if item['type'] == 'query'),
                     None)

        with st.container():
            st.markdown("**🙋 Your Question:**")
            st.info(query)

        st.markdown("---")

        # Display all answers with feedback
        answers = [item for item in st.session_state.conversation_history if item['type'] == 'answer']
        feedbacks = [item for item in st.session_state.conversation_history if item['type'] == 'feedback']

        for i, answer in enumerate(answers):
            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"**🤖 AI Answer (Iteration {answer['iteration']}):**")

            with col2:
                iteration_badge = f"Iteration {answer['iteration']}"
                if answer['iteration'] == 1:
                    st.success(iteration_badge)
                else:
                    st.info(iteration_badge)

            st.write(answer['content'])

            # Show feedback for this iteration if exists
            if i < len(feedbacks):
                with st.expander(f"💬 Feedback for Iteration {answer['iteration']}"):
                    st.warning(feedbacks[i]['content'])

            st.markdown("---")

        # Current state info
        current_state = st.session_state.current_state

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Iteration", current_state['iteration_count'])
        with col2:
            st.metric("Max Iterations", current_state['max_iterations'])
        with col3:
            remaining = current_state['max_iterations'] - current_state['iteration_count']
            st.metric("Remaining", remaining)

        st.markdown("---")

        # Feedback section
        if not current_state['user_satisfied'] and current_state['iteration_count'] < current_state['max_iterations']:
            st.markdown("### 💭 Provide Feedback")

            col1, col2 = st.columns([3, 1])

            with col1:
                feedback = st.text_area(
                    "How can we improve this answer?",
                    height=100,
                    placeholder="e.g., 'Add more examples', 'Simplify the explanation', 'Include code samples'"
                )

            with col2:
                st.markdown("**Quick Actions:**")
                if st.button("✅ I'm Satisfied", use_container_width=True, type="primary"):
                    current_state['user_satisfied'] = True
                    st.session_state.current_state = current_state
                    st.balloons()
                    st.success("🎉 Great! Conversation completed!")
                    st.rerun()

            if st.button("🔄 Improve Answer", use_container_width=True):
                if feedback.strip():
                    with st.spinner("🤖 AI is refining the answer..."):
                        # Update state with feedback
                        current_state['feedback_history'].append(feedback)
                        current_state['user_satisfied'] = False

                        # Add feedback to history
                        st.session_state.conversation_history.append({
                            "type": "feedback",
                            "content": feedback
                        })

                        # Check if should continue
                        decision = check_user_satisfied(current_state)

                        if decision == "continue":
                            # Generate improved answer
                            app = create_graph()
                            improved_state = improve_answer(current_state)

                            # Update session state
                            st.session_state.current_state = improved_state
                            st.session_state.conversation_history.append({
                                "type": "answer",
                                "content": improved_state['current_answer'],
                                "iteration": improved_state['iteration_count']
                            })

                            st.success("✅ Answer improved!")
                            st.rerun()
                        else:
                            st.warning("⚠️ Max iterations reached!")
                else:
                    st.warning("⚠️ Please provide feedback to improve the answer!")

        else:
            # Final answer section
            st.markdown("### 🎯 Final Answer")

            if current_state['user_satisfied']:
                st.success("✅ You marked this answer as satisfactory!")
            else:
                st.info("ℹ️ Maximum iterations reached.")

            # Final answer display
            with st.container():
                st.markdown("**🏆 Final Approved Answer:**")
                st.success(current_state['current_answer'])

            # Summary
            st.markdown("---")
            st.markdown("### 📊 Summary")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Iterations", current_state['iteration_count'])
            with col2:
                st.metric("Feedback Rounds", len(current_state['feedback_history']))
            with col3:
                status = "Satisfied ✅" if current_state['user_satisfied'] else "Max Reached ⚠️"
                st.metric("Status", status)

            # Download button for final answer
            st.download_button(
                label="💾 Download Final Answer",
                data=current_state['current_answer'],
                file_name="ai_response.txt",
                mime="text/plain",
                use_container_width=True
            )


if __name__ == "__main__":
    main()