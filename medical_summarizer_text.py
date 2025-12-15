from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
import gradio as gr

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


def summarize_report(user_report):
    if not user_report.strip():
        return "Please enter a medical report to summarize."

    response = summarizer_chain.invoke({'report': user_report})
    return response.content


# Gradio Interface:

with gr.Blocks() as app:
    gr.Markdown('## Medical Report Summarizer')
    gr.Markdown('Please paste your medical report and get a patient-friendly summary')

    report_input = gr.Textbox(lines=12, label='Medical Report', placeholder='Paste your medical report here...')
    sum_button = gr.Button('Summarize the report')
    report_output = gr.Textbox(lines=12, label='Summary for Patient')

    sum_button.click(
        summarize_report,
        inputs=report_input,
        outputs=report_output
    )

app.launch()