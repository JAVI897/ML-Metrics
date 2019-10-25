"""Home page shown when the user enters the application"""
import streamlit as st
import numpy as np
import os

data = []
def write():
    """Method used to write page in app.py"""
    st.title("ML-Metrics")
    number_models = st.slider("Number of models", min_value = 1, max_value = 5, value = 1)
    
    def file_selector(folder_path='./data'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a file', filenames)
        return selected_filename
    
    def confirm(selected_gt, selected_pred):
            filenames = os.listdir('./data')
            return selected_gt in filenames and selected_pred in filenames
        
    global data
    data = []
    
    
    if number_models == 1:
        
        
        st.subheader("Model: ground truth")
        selected_gt = st.text_input('Name of ground truth')
        st.text(selected_gt)
        
        st.subheader("Model: prediction")
        selected_pred = st.text_input('Name of prediction')
        st.text(selected_pred)
        w5 = st.button("Confirm")
        
        if w5:
            if confirm(selected_gt, selected_pred):
                st.success('Great!')
                st.balloons()
                data.append((np.load("data/" + selected_gt), np.load("data/" + selected_pred)))

            else:
                st.error('Filename not found!')
        
    
    #data.append((np.load( gt_model), np.load(predict_model)))