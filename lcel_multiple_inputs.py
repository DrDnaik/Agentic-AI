from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv() # env
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini',temperature=1)

def chain_def():

    prompt = PromptTemplate.from_template('Tell me joke about {topic} in {style} format')
    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain

def run_chain(user_topic,user_style):
    return chain_def().invoke({'topic':user_topic,'style':user_style})

result = run_chain('medicine','poem')

print('Result : ',result)