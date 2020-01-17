"""Curve type page"""
import streamlit as st
import pandas as pd
import metrics
import numpy as np
import src.st_extensions
from src.pages.home import data
import plotly.graph_objects  as go
from src.pages.pages_functions import configuration, methods_prc, methods_roc, methods_both, grafico

def write():
    """Method used to write page in app.py"""
    
    if len(data) == 0:
        st.error("There's not data")
    
    else:
        df_curves=pd.DataFrame({"curve":["ROC Curve","PRC Curve","Both"]})

        #Habría que cambiarlo de momento añadimos datos por aquí
        #prediction_1=np.load("data/prediction_1.npy")
        #Y_Test_1=np.load("data/Y_Test_1.npy")
        #data = [(Y_Test_1, prediction_1)]
        keys = list(data.keys())
        model = st.multiselect(label="Select models:", options = keys, default = [keys[0]] )
        metric_data = [v for k, v in data.items() if k in model]
        model_names = [k for k, v in data.items() if k in model]
        graphs = metrics.Graphs(metric_data, model_names)
        
        st.sidebar.title("Curve type")
        option_curve= st.sidebar.selectbox("Select",df_curves["curve"])
        option_threshold, option_fill, option_legend, number_threshold= configuration()

        if option_curve == "ROC Curve": 

            st.title("ROC Curve")
            if option_threshold: 
                methods_list_roc= methods_roc() 
            else:
                methods_list_roc=None
            g = grafico(graphs, option_curve, option_threshold,option_fill,option_legend,methods_list_roc, number_threshold)
            st.plotly_chart(g)
            #st.plotly_chart(g,  width=702, height=900)


        elif option_curve == "PRC Curve": 

            st.title("PRC Curve")  
            if option_threshold: 
                methods_list_prc= methods_prc()
            else: 
                methods_list_prc=None
            g = grafico(graphs, option_curve, option_threshold,option_fill,option_legend,methods_list_prc, number_threshold)
            st.plotly_chart(g)
            #st.plotly_chart(g, width=702, height=900)

        elif option_curve == "Both":
            st.title("ROC and PRC curves")
            if option_threshold: 
                methods_list_both = methods_both()
            else:
                methods_list_both = None
            g = grafico(graphs, option_curve, option_threshold,option_fill,option_legend,methods_list_both, number_threshold)
            st.plotly_chart(g)
            #st.plotly_chart(g, height=670, width=770)
        
        
        
