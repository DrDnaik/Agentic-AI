from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

load_dotenv()  # env
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

print(conversation.predict(input="Hello, my name is Alex."))
print(conversation.predict(input="Can you remind me what my name is?"))
print(conversation.predict(input="What did I just ask you?"))

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

load_dotenv()  # env
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Replace hardcoded strings with a loop for continuous user input
print("AI Chatbot is ready. Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("AI Chatbot: Goodbye!")
        break

    # Predict the response based on the user's input
    ai_response = conversation.predict(input=user_input)
    print(f"AI Chatbot: {ai_response}")
