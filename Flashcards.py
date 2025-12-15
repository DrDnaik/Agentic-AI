import os
from dotenv import load_dotenv
from  langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import gradio as gr
from bs4 import BeautifulSoup
import requests


load_dotenv()
import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

flashcard_prompt = PromptTemplate(
    input_variables=["text", "num_cards"],
    template="""
    You are a teacher creating flashcards for revision. 
    Given the following study material, generate {num_cards} flashcards(no questions just the information as flashcards)  

    Study material: {text}
    """
)

flashcard_chain = flashcard_prompt | llm

def extract_text_from_url(url):

    try:

        response = requests.get(url, timeout=10)

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script/style tags

        for script in soup(["script", "style"]):

            script.extract()

        text = soup.get_text(separator=" ")

        text = " ".join(text.split())  # Clean extra spaces

        return text[:5000]  # limit to avoid token overload

    except Exception as e:

        return f"Error fetching URL: {e}"


def generate_flashcards_url(url, num_cards=5):
    text= extract_text_from_url(url)
    response = flashcard_chain.invoke({"text": text, "num_cards": num_cards})
    return response.content


with gr.Blocks() as demo:
    gr.Markdown("##  LLM-Powered Flashcards App ")
    gr.Markdown("Upload notes/text → Generate Flashcards")
    url_input = gr.Textbox(

      placeholder="Paste your webpage url..."
    )
    num_cards = gr.Slider(3, 15, value=5, step=1, label="Number of Flashcards")
    generate_btn = gr.Button("Generate Flashcards")
    flashcards_output = gr.Textbox(label="Flashcards", lines=10)
    generate_btn.click(

        generate_flashcards_url,

        inputs=[url_input, num_cards],

        outputs=flashcards_output

    )

demo.launch()







