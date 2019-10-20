"""Other graphs page"""
import streamlit as st
import pandas as pd
import metrics
import numpy as np
import src.st_extensions
from src.pages.pages_functions import configuration, methods_Precision_and_Recall, other_graphics, methods_tprate_and_fprate

def write():
    """Method used to write page in app.py"""
    
    df_graphs=pd.DataFrame({"graphs":["Precision and Recall vs Decision threshold", 
                                      "True positive rate and False positive rate vs Decision threshold"]})
    
    #Sensitivity and 1 - specificity vs Decision threshold
    
    #Habría que cambiarlo de momento añadimos datos por aquí
    prediction_1=np.load("data/prediction_1.npy")
    Y_Test_1=np.load("data/Y_Test_1.npy")
    data = [(Y_Test_1, prediction_1)]
    graphs = metrics.Graphs(data)
    
    st.sidebar.title("Other graphs")
    option_graphs = st.sidebar.selectbox("Select",df_graphs["graphs"])
    option_threshold,option_fill,option_legend,number_threshold= configuration(other_graph = True)
    
    if option_graphs == "Precision and Recall vs Decision threshold":
        st.title("Precision and Recall vs Decision threshold")
        if option_threshold: 
            methods_list = methods_Precision_and_Recall()
        else: 
            methods_list=None
        g = other_graphics(graphs, option_graphs, option_threshold,option_legend,methods_list, number_threshold)
        st.plotly_chart(g, width=702, height=900)
        
    elif option_graphs == "True positive rate and False positive rate vs Decision threshold":
        st.title("True positive rate and False positive rate vs Decision threshold")
        if option_threshold: 
            methods_list = methods_tprate_and_fprate()
        else: 
            methods_list=None
        g = other_graphics(graphs, option_graphs, option_threshold,option_legend,methods_list, number_threshold)
        st.plotly_chart(g, width=702, height=900)
        
        
        
        