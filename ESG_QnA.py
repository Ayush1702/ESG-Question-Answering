import re
import copy
import numpy as np
from transformers import AutoConfig, AutoModel, QuestionAnsweringPipeline
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
from numpy import ndarray
from transformers import pipeline
import streamlit as st

st.title("ESG Question Answering")
st.write('You can paste your context text in the field below along with the question.')

HTML_WRAPPER = """<div style="
                       overflow-x: auto; 
                       border: 1px solid #e6e9ef; 
                       border-radius: 0.25rem; 
                       padding: 1rem; 
                       margin-bottom: 
                       2.5rem">
                       {}</div>"""

def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(https://unsplash.com/photos/jqgsM3B9Fpo/download?force=true);
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack_url()

@st.cache(allow_output_mutation=True)
def esg_question_answering():
    model_name = "roberta-base/"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, local_files_only=True)
    return model

with open("sample_context.txt", encoding="utf-8") as f:
    contents = f.readlines()
    sample_input_context = "".join(contents).lstrip().replace("\n"," ").replace("  ", " ")
session_question_input_sample = "What is the emission reduction target aimed?"

context_para_input = st.text_area("Enter Context Paragraph", sample_input_context, height = 400)
question_input = st.text_area("Enter Question", session_question_input_sample, height = 25)
st.session_state.context, st.session_state.question_default = context_para_input, question_input

if st.button('Submit'):
    context_input = st.session_state.context
    question_input = st.session_state.question_default
    with st.spinner('Loading Model'):
        esg_model = esg_question_answering()
    tokenizer_path = "/app/esg-question-answering/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    question_answerer = pipeline("question-answering", model=esg_model, tokenizer=tokenizer)
    result = question_answerer(question=question_input, context=context_input)
    st.write(HTML_WRAPPER.format(result['answer']), unsafe_allow_html=True)