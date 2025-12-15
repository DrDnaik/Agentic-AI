from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
import gradio as gr
import PyPDF2

'''
PDF --> Bytes [0,1] --> PyPDF2 PDF Reader
'''

load_dotenv()  # env
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)

summarizer_prompt = PromptTemplate(
    input_variables=['report'],
    template="""
    You are a medical expert. Summarize the following medical report in simple terms, clear language for the patient : 

    Medical Report : 
    {report}

    Summary (for patient) : 

"""
)

summarizer_chain = summarizer_prompt | llm


# variable  : report  --> Call this summarizer_chain

def summarize_report(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file.name)
    text = ''
    for page in pdf_reader.pages:
        text = text + page.extract_text()
    response = summarizer_chain.invoke({'report': text})
    return response.content


# Gradio :

with gr.Blocks() as app:
    gr.Markdown('## Medical Report Summarizer')
    gr.Markdown('Please past your medical report and get a patient-friendly summary')

    report_input = gr.File(label='Medical Report', file_types=['.pdf'])
    sum_button = gr.Button('Summarize the report')
    report_output = gr.Textbox(lines=12, label='Summary for Patient')

    sum_button.click(
        summarize_report,
        inputs=report_input,
        outputs=report_output
    )

app.launch()
