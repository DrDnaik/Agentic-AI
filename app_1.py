from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
prompt = ChatPromptTemplate.from_template("Tell me a joke about {input}")
parser = StrOutputParser()


chain = prompt | llm | parser

# Run
result = chain.invoke({"input": "AI"})
print("Result:", result)
