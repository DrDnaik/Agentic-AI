from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()  # env
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)

'''
1. User will be giving their symptoms (get from user)

2. I need to decide - Blood Test is required or not  [ Conditional Statement ] [ Classfication ]

3. If Blood Test is required - Why it is required and not home remedy?

4. If Blood Test is not required - Home remedy is fine - Why Home remedy will be helpful ? 

'''

'''
1. Classifier 
'''

classifier_prompt = PromptTemplate.from_template("""

Symptoms : {symptoms}

Decision : Blood Test is required or not

Given the symptoms, decide whether Blood Test is required or not. 

If symptoms show a sign of infection, return "NEEDS_TEST"

If symptoms are mild, and does not required blood test, return "NO_TEST"

Return only "NEEDS_TEST" or "NO_TEST" - Nothing else. Just one word.                                                                                                                                   

"""
                                                 )

classifier_chain = classifier_prompt | llm | StrOutputParser()

'''
2. Why we need test 
'''

needs_test_prompt = PromptTemplate.from_template("""

Symptoms : {symptoms}

Decision : Blood Test is required

Given the symptoms, and given the decision : blood test is required : 

Explain in detail why blood test is required and why not home remedy will work                                                                                                                                  

"""
                                                 )

'''
3. home remedy
'''
needs_no_test_prompt = PromptTemplate.from_template("""

Symptoms : {symptoms}

Decision : Home remedy is fine. No blood test is required

Given the symptoms, and given the decision : Home remedy is fine. No blood test is required : 

Explain in detail why blood test is not required and why home remedy will work                                                                                                                                  

"""
                                                    )


def conditional_logic(input_data):
    symptoms = input_data['symptoms']
    decision = classifier_chain.invoke({'symptoms': symptoms})
    if decision == 'NEEDS_TEST':
        final_prompt = needs_test_prompt.format(symptoms=symptoms)
    else:
        final_prompt = needs_no_test_prompt.format(symptoms=symptoms)
    return final_prompt


conditional_chain = RunnableLambda(conditional_logic) | llm | StrOutputParser()

input = input('how are you feeling ? ')

result = conditional_chain.invoke({'symptoms': input})

print('Result : \n\n', result)