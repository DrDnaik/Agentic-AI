from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
import gradio as gr
import pandas as pd  # <-- use pandas for Excel

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

# Initialize LLM
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)

# Prompt template
summarizer_prompt = PromptTemplate(
    input_variables=['report'],
    template="""
    You are a medical expert. Summarize the following medical report in simple, clear language for the patient.

    Medical Report:
    {report}

    Summary (for patient):
    """
)

summarizer_chain = summarizer_prompt | llm


# ✅ Function to summarize Excel file
def summarize_report(excel_file):
    # Read Excel file (requires openpyxl)
    df = pd.read_excel(excel_file.name)

    # Combine all text data from Excel into one string
    text = df.to_string(index=False)

    # Pass to LLM summarizer
    response = summarizer_chain.invoke({'report': text})

    return response.content


# Optional: Gradio UI (if you want to use it)
def app():
    with gr.Blocks() as demo:
        gr.Markdown("### 🩺 Medical Report Summarizer (Excel Version)")
        excel_input = gr.File(label="Upload Excel File (.xlsx)")
        output = gr.Textbox(label="Summary", lines=10)
        btn = gr.Button("Summarize")

        btn.click(fn=summarize_report, inputs=excel_input, outputs=output)
    return demo


if __name__ == "__main__":
    app().launch()
