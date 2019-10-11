"""Home page shown when the user enters the application"""
import streamlit as st
import numpy as np
import os

data = None

def write():
    """Method used to write page in app.py"""
    st.title("ML-Metrics")
    number_models = None
    number_models = st.text_input("Number of models")
    
    def file_selector(folder_path='./data'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a file', filenames)
        return selected_filename
    
    global data
    data = []
    
    if number_models != None:
        number_models = int(number_models)
        for i in range(number_models):
            st.header("Model: {}".format(i+1))

            st.subheader("Prediction: ")
            predict_model = file_selector()
            st.write('You selected `%s`' % predict_model)

            st.subheader("Ground truth: ")
            gt_model = file_selector()
            st.write('You selected `%s`' % gt_model)
            
            #data.append((np.load( gt_model), np.load(predict_model)))