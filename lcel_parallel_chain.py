from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel
import os

load_dotenv()  # env
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini', temperature=1)

'''
Given a symptom : 

--> I want the cause for the symptom : cause_prompt
--> I want the medicines for the symptom : medicine_prompt
--> How can I prevent it in my future?  : prevention_prompt

'''


def parallel_chain():
    cause_prompt = PromptTemplate.from_template(
        "Given the symptom : {symptom} , Can you please explain what are its causes ? "
    )

    medicine_prompt = PromptTemplate.from_template(
        "Given the symptom : {symptom} , Please suggest medicines for it! "
    )

    prevention_prompt = PromptTemplate.from_template(
        "I am having this symptom : {symptom} , Can you please suggest how to prevent it in future? "
    )

    parallel_chain = RunnableParallel(
        cause_chain=cause_prompt | llm | StrOutputParser(),
        medicine_chain=medicine_prompt | llm | StrOutputParser(),
        prevention_chain=prevention_prompt | llm | StrOutputParser())

    return parallel_chain


def run_chain(input):
    pc = parallel_chain()
    result = pc.invoke({'symptom': input})
    return result


symptom = input('Enter your symptom : ')
result = run_chain(symptom)

formatted_result = f"""

1. Cause of the {symptom} : 

{result['cause_chain']}



2. Treatment of the {symptom} : 

{result['medicine_chain']}

3. Prevention of the {symptom} : 

{result['prevention_chain']}


"""

print(formatted_result)