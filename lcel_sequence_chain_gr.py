from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os
import gradio as gr

load_dotenv() # env
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

llm = ChatOpenAI(model='gpt-4o-mini',temperature=0.2)

'''
1. Lets generate a random topic in medicine  -- 1st Chain (topic_prompt | llm | parser) : topic_chain
2. Then, it has to write an article about it  -- 2nd Chain  (article_prompt | llm | parser) : article_chain
'''

def sequence_chain():

    topic_prompt = PromptTemplate.from_template(
        "Generate a unique {subject} topic in one word"
    )
    topic_chain = topic_prompt | llm | StrOutputParser()


    # --------

    article_prompt = PromptTemplate.from_template(
        "Write a short paragraph about {topic}"
    )
    article_chain = article_prompt | llm | StrOutputParser()

    final_chain = topic_chain | article_chain

    return final_chain

def run_sequence_chain(user_subject):
    chain = sequence_chain()
    result = chain.invoke({'subject':user_subject})
    return result

# Gradio Interface:

with gr.Blocks() as app:
    gr.Markdown('## Article Generator')
    gr.Markdown('Please enter the topic to generate article on it')

    topic_input = gr.Textbox(lines=6, label='Topic')

    sum_button = gr.Button('Generate article')
    topic_output = gr.Textbox(lines=12, label='Article')

    sum_button.click(
        run_sequence_chain,
        inputs=topic_input,
        outputs=topic_output
    )

app.launch()