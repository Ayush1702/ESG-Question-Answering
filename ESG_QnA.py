import re
import os
import copy
import numpy as np
from transformers import AutoConfig, AutoModel, RobertaForQuestionAnswering, AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
from numpy import ndarray
import streamlit as st
cwd = os.getcwd()
st.text(cwd)
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
             background: url(https://unsplash.com/photos/aL7SA1ASVdQ/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MTl8fGZvcmVzdHxlbnwwfHx8fDE2NjExNTAyNTY&force=true);
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack_url()

st.sidebar.header("Select Examples")
add_selectbox = st.sidebar.selectbox(
    "Load preset context example",
    ("Example 1 - Ventas Inc", "Example 2 - Bridgestone Corporation", "Example 3 - Humana Inc")
)

question_selectbox = st.sidebar.selectbox(
    "Load preset question",
    ("Target Aimed", "Methodology/Mechanism")
)

def esg_question_answering():
    parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))
    model_dir = os.path.join(parent_dir, "esg-question-answering/roberta-base/")
    model_name = model_dir
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, local_files_only=True)
    return model

if add_selectbox == 'Example 1 - Ventas Inc':
    with open("sample_context.txt", encoding="utf-8") as f:
        contents = f.readlines()
        sample_input_context = "".join(contents).lstrip().replace("\n"," ").replace("  ", " ")
    if question_selectbox == 'Target Aimed':
        session_question_input_sample = "What is the emission reduction target supposedly aimed?"
    elif question_selectbox == 'Methodology/Mechanism':
        session_question_input_sample = "What emssion reduction methodology or mechanism is used here?"

    context_para_input = st.text_area("Enter Context Paragraph", sample_input_context, height = 400)
    question_input = st.text_area("Enter Question", session_question_input_sample, height = 15)
    st.session_state.context, st.session_state.question_default = context_para_input, question_input

if add_selectbox == 'Example 2 - Bridgestone Corporation':
    with open("sample_context_2.txt", encoding="utf-8") as f:
        contents = f.readlines()
        sample_input_context = "".join(contents).lstrip().replace("\n"," ").replace("  ", " ")
    if question_selectbox == 'Target Aimed':
        session_question_input_sample = "What is the emission reduction target supposedly aimed?"
    elif question_selectbox == 'Methodology/Mechanism':
        session_question_input_sample = "What emssion reduction methodology or mechanism is used here?"

    context_para_input = st.text_area("Enter Context Paragraph", sample_input_context, height = 400)
    question_input = st.text_area("Enter Question", session_question_input_sample, height = 15)
    st.session_state.context, st.session_state.question_default = context_para_input, question_input

if add_selectbox == 'Example 3 - Humana Inc':
    with open("sample_context_3.txt", encoding="utf-8") as f:
        contents = f.readlines()
        sample_input_context = "".join(contents).lstrip().replace("\n"," ").replace("  ", " ")
    if question_selectbox == 'Target Aimed':
        session_question_input_sample = "What is the emission reduction target supposedly aimed?"
    elif question_selectbox == 'Methodology/Mechanism':
        session_question_input_sample = "What emssion reduction methodology or mechanism is used here?"

    context_para_input = st.text_area("Enter Context Paragraph", sample_input_context, height = 400)
    question_input = st.text_area("Enter Question", session_question_input_sample, height = 15)
    st.session_state.context, st.session_state.question_default = context_para_input, question_input

if st.button('Submit'):
    context_input = st.session_state.context
    question_input = st.session_state.question_default
    esg_model = esg_question_answering()
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", local_files_only=True)
    question_answerer = pipeline("question-answering", model=esg_model, tokenizer=tokenizer, framework="pt")
    result = question_answerer(question=question_input, context=context_input)
    if result['answer'] == '.' or '':
        st.text("no answer present in the given context")
    else:
        st.write(HTML_WRAPPER.format(result['answer']), unsafe_allow_html=True)