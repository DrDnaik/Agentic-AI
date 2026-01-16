import streamlit as st
import os
import asyncio
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load env vars
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="SQL MCP Chatbot", page_icon="💬", layout="centered")
st.title("💬 SQL MCP Database Assistant")
st.markdown("Ask me anything about your database (via MCP tools).")

if not OPENAI_API_KEY:
    st.error("⚠️ Please set OPENAI_API_KEY in .env file")
    st.stop()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# Async tool loader (cached)
@st.cache_resource
def load_mcp_tools():
    async def _load():
        mcp_server_path = os.path.join(os.path.dirname(__file__), "mcp_server.py")

        client = MultiServerMCPClient(
            {
                "sql_db": {
                    "command": "python",
                    "args": [mcp_server_path],
                    "transport": "stdio"
                }
            }
        )
        tools = await client.get_tools()
        return client, tools

    try:
        return asyncio.run(_load())
    except Exception as e:
        st.error(f"Failed to load MCP tools: {str(e)}")
        st.error("Make sure mcp_server.py is working correctly.")
        st.stop()


try:
    client, mcp_tools = load_mcp_tools()
except Exception as e:
    st.error(f"Error initializing MCP client: {str(e)}")
    st.stop()

SYSTEM_PROMPT = """
You are a helpful SQL database assistant.
You have access to MCP tools that let you:
1. List all tables in the database
2. Describe the schema of any table
3. Run SELECT queries to retrieve data

Rules:
- Always inspect schema before writing queries
- Use ONLY safe SELECT queries
- Explain results in plain English
"""

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about your database..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Bind MCP tools to LLM
            llm_with_tools = st.session_state.llm.bind_tools(mcp_tools)

            # Prepare chat messages
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(st.session_state.messages[-10:])

            # Create a expander for showing the thinking process
            thinking_expander = st.expander("View Thinking Process", expanded=False)
            step_counter = 1

            # Initial API call : will be having only tools , no content
            response = llm_with_tools.invoke(messages)

            # If there is tool calls, process it :

            while getattr(response, "tool_calls", None):

                messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.tool_calls
                })

                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']

                    with thinking_expander:
                        st.markdown(f"**Step {step_counter}: Calling `{tool_name}`**")

                        if tool_args:
                            st.code(json.dumps(tool_args, indent=2), language="json")
                        else:
                            st.info("No arguments")

                    # Execute the tool :
                    try:

                        # Find the tool one by one
                        tool = next(t for t in mcp_tools if t.name == tool_name)


                        # Invoke the tool
                        async def invoke_tool():
                            return await tool.ainvoke(tool_args)


                        tool_result = asyncio.run(invoke_tool())
                        tool_result_str = str(tool_result)

                        with thinking_expander:
                            st.success("Result : ")
                            st.code(tool_result_str, language="text")
                            st.markdown("---")

                    except Exception as e:

                        tool_result_str = 'Error in the tool'
                        with thinking_expander:
                            st.error("Some error in the tool", str(e))

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call['id'],
                        "name": tool_name,
                        "content": tool_result_str
                    })

                    step_counter += 1

                response = llm_with_tools.invoke(messages)
                # Display the final response :

                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})


















