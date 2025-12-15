from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv() # env
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini',temperature=0.2)

'''
1. Prescription - Extract info from that : prompt
2. Make it easy - more detailed guidlines
'''

prescription_prompt = PromptTemplate.from_template('''
You are a Prescription Interpreter. Given the doctor's notes , extract : 

Medication Name : 
Dosage : 
When to take : 
Duration :                                                                                                                                        
Special Instructions (if any) : 

Doctor's Prescription : {prescription}                                            
'''
)

extract_chain = prescription_prompt | llm | StrOutputParser()

make_it_easy_for_arvind = PromptTemplate.from_template('''
Convert the medical prescription into a clear daily schedule : 

Make it easy enough so that even a 8th grader can follow 

Format :


- Morning  
- Afternoon
- EVening 
- Night 

Additional Instructions : 


Prescription : {prescription_info}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
'''
)

make_it_easy_for_arvind_chain = make_it_easy_for_arvind | llm | StrOutputParser()

medical_chain = (extract_chain | (lambda prescription:{'prescription_info': prescription} ) | make_it_easy_for_arvind_chain)

input = '''
Tab Amoxicillin 500mg – One tablet three times daily for 5 days.
Take after meals.
Tab Paracetamol 500mg – Only if fever, every 6 hours if needed.
'''

result = medical_chain.invoke({'prescription':input})

print(result)

# Patient Name : {patient_name}
# '''
# )
# make_it_easy_for_arvind_chain = make_it_easy_for_arvind | llm | StrOutputParser()
# medical_chain = (extract_chain | (lambda extracted: {
#         "prescription_info": extracted,
#         "patient_name": "Arvind"
#       }) | make_it_easy_for_arvind_chain)