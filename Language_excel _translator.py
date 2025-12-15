from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
import gradio as gr
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_transformers.doctran_text_translate import DoctranTextTranslator

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

# Initialize LLM
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
# Translation prompt

translate_prompt = PromptTemplate(
    input_variables=['text', 'source_lang', 'target_lang'],
    template="""
    You are a professional translator.
    Translate the following text from {source_lang} to {target_lang}.
    Keep the tone, context, and meaning accurate.

    Text:
    {text}

    Translation:
    """
)

# Load Excel file
df = pd.read_excel('your_excel_file.xlsx')

# Save as CSV
df.to_csv('your_excel_file.csv', index=False)

loader = CSVLoader(file_path='your_excel_file.csv')
documents = loader.load()

# Initialize the translator with your OpenAI API key and target language
translator = DoctranTextTranslator(openai_api_key="YOUR_OPENAI_API_KEY",
                                   language="target_language_code")  # e.g., 'es' for Spanish

# Translate the documents
translated_documents = translator.transform_documents(documents)

# Example: Extract translated content and put it in a list
translated_texts = [doc.page_content for doc in translated_documents]

# Example: Convert back to a DataFrame and save as Excel
translated_df = pd.DataFrame({'Translated_Content': translated_texts})
translated_df.to_excel('translated_excel_file.xlsx', index=False)