import re
import copy
import numpy as np
from transformers import AutoConfig, AutoModel, QuestionAnsweringPipeline
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from transformers import RobertaTokenizer, RobertaForQuestionAnswering, RobertaConfig
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

# @st.cache(allow_output_mutation=True)
# def esg_question_answering():
    # model_name = "/app/esg-question-answering/roberta-base"
    # # config = AutoConfig.from_pretrained(model_name, cache_dir= model_name)
    # model = RobertaForQuestionAnswering.from_pretrained(model_name)
    # # config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
    # # model = AutoModelForQuestionAnswering.from_config(config)
    # return model

# if 'context' not in st.session_state:
    # st.session_state.context = None
# if 'question_default' not in st.session_state:
    # st.session_state.question_default = None

with open("sample_context.txt", encoding="utf-8") as f:
    contents = f.readlines()
    sample_input_context = "".join(contents).lstrip().replace("\n"," ").replace("  ", " ")
session_question_input_sample = "What is the emission reduction target aimed?"

context_para_input = st.text_area("Enter Context Paragraph", sample_input_context, height = 400)
question_input = st.text_area("Enter Question", session_question_input_sample, height = 25)
st.session_state.context, st.session_state.question_default = context_para_input, question_input
proxies = {
  "http": "http://10.10.1.10:3128",
  "https": "https://10.10.1.10:1080",
}
if st.button('Submit'):
    context_input = st.session_state.context
    question_input = st.session_state.question_default
    model_name = "/app/esg-question-answering/roberta-base/"
    with st.spinner('Loading Model'):
        config = RobertaConfig.from_pretrained(model_name, cache_dir= model_name, proxies=proxies)
        esg_model = RobertaForQuestionAnswering.from_config("/app/esg-question-answering/roberta-base/")
    tokenizer_path = "/app/esg-question-answering/roberta-base/"
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path, proxies=proxies)
    question_answerer = pipeline("question-answering", model=esg_model, tokenizer=tokenizer)
    result = question_answerer(question=question_input, context=context_input)
    st.write(HTML_WRAPPER.format(result['answer']), unsafe_allow_html=True)

# Make predictions with the model
# to_predict = [
    # {
        # "context": context_para_input,
        # "qas": [
            # {
                # "question": "What is the emission reduction mechanism or technology used here?",
                # "id": "0",
            # }
        # ],
    # }
# ]

