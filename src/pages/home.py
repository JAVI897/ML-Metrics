"""Home page shown when the user enters the application"""
import streamlit as st
import numpy as np
import os

data = None

def write():
    """Method used to write page in app.py"""
    st.title("ML-Metrics")
    number_models = None
    number_models = st.text_input("Number of models", value="0")
    
    def file_selector(folder_path='./data'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a file', filenames)
        return selected_filename
    
    global data
    data = []
    
    #data.append((np.load( gt_model), np.load(predict_model)))