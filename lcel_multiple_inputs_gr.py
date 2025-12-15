from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os
import gradio as gr

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

# Gradio Interface:

with gr.Blocks() as app:
    gr.Markdown('## Joke Generator')
    gr.Markdown('Please enter the topic and style to generate joke on the topic in a unique way')

    topic_input = gr.Textbox(lines=6, label='Topic', placeholder='medicine,AI,education etc')
    style_input=gr.Textbox(lines=6, label='Style', placeholder='poem,formal,funny')
    sum_button = gr.Button('Generate joke')
    topic_output = gr.Textbox(lines=12, label='Joke')

    sum_button.click(
        run_chain,
        inputs=[topic_input,style_input],
        outputs=topic_output
    )

app.launch()