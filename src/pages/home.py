"""Home page shown when the user enters the application"""
import streamlit as st
import numpy as np
import os
import collections

data = []
def write():
    """Method used to write page in app.py"""
    st.title("ML-Metrics")
    number_models = st.slider("Number of models", min_value = 1, max_value = 5, value = 1)
        
    global data
    data = collections.OrderedDict()
    if number_models >= 1:
        st.subheader("Model 1")
        model1 = st.text_input('Name of model 1')
        selected_gt1 = st.file_uploader(label="Model: ground truth", type=['npy'], key=0)
        selected_pred1 = st.file_uploader(label="Model: prediction", type=['npy'], key=1)
        
        if number_models >= 2:
            st.subheader("Model 2")
            model2 = st.text_input('Name of model 2')
            selected_gt2 = st.file_uploader(label="Model: ground truth", type=['npy'], key=2)
            selected_pred2 = st.file_uploader(label="Model: prediction", type=['npy'], key=3)
            
            if number_models >=3:
                st.subheader("Model 3")
                model3 = st.text_input('Name of model 3')
                selected_gt3 = st.file_uploader(label="Model: ground truth", type=['npy'], key=4)
                selected_pred3 = st.file_uploader(label="Model: prediction", type=['npy'], key=5)
                
                if number_models >= 4:
                    st.subheader("Model 4")
                    model4 = st.text_input('Name of model 4')
                    selected_gt4 = st.file_uploader(label="Model: ground truth", type=['npy'], key=6)
                    selected_pred4 = st.file_uploader(label="Model: prediction", type=['npy'], key=7)
                    
                    if number_models >= 5:
                        st.subheader("Model 5")
                        model5 = st.text_input('Name of model 5')
                        selected_gt5 = st.file_uploader(label="Model: ground truth", type=['npy'], key=8)
                        selected_pred5 = st.file_uploader(label="Model: prediction", type=['npy'], key=9)
                
    w5 = st.button("Confirm")
    
    if w5:
            
        st.success('Great!')
        st.balloons()
        if number_models == 1:
            data[model1] = (np.load(selected_gt1), np.load(selected_pred1))
        elif number_models == 2:
            data[model1] = (np.load(selected_gt1), np.load(selected_pred1))
            data[model2] = (np.load(selected_gt2), np.load(selected_pred2))
            
        elif number_models == 3:
            data[model1] = (np.load(selected_gt1), np.load(selected_pred1))
            data[model2] = (np.load(selected_gt2), np.load(selected_pred2))
            data[model3] = (np.load(selected_gt3), np.load(selected_pred3))
        
        elif number_models == 4:
            data[model1] = (np.load(selected_gt1), np.load(selected_pred1))
            data[model2] = (np.load(selected_gt2), np.load(selected_pred2))
            data[model3] = (np.load(selected_gt3), np.load(selected_pred3))
            data[model4] = (np.load(selected_gt4), np.load(selected_pred4))
        
        else:
            data[model1] = (np.load(selected_gt1), np.load(selected_pred1))
            data[model2] = (np.load(selected_gt2), np.load(selected_pred2))
            data[model3] = (np.load(selected_gt3), np.load(selected_pred3))
            data[model4] = (np.load(selected_gt4), np.load(selected_pred4))
            data[model5] = (np.load(selected_gt5), np.load(selected_pred5))