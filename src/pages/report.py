"""Report page"""
import streamlit as st
import pandas as pd
import metrics
import numpy as np
import src.st_extensions
import plotly.graph_objects  as go
from src.pages.pages_functions import configuration_report, pie_chart

def write():
    """Method used to write page in app.py"""
    
    #Habría que cambiarlo de momento añadimos datos por aquí
    prediction_1=np.load("data/prediction_1.npy")
    Y_Test_1=np.load("data/Y_Test_1.npy")
    
    
    df_methods_pie_chart= pd.DataFrame({"method":["Youden","F-score","Distance_ROC",
                                              "Distance_PRC","Difference_Sensitivity_Specificity",
                                              "Difference_Recall_Precision"]})
    
    st.title("Report")
    st.sidebar.title("Report")
    colormap = configuration_report()
    optimum= metrics.Optimum(Y_Test_1,prediction_1)
    r= optimum.report(colormap=colormap)
    st.dataframe(r)
    
    
    r_pie_=optimum.report(colormap=False)
    st.title("Confusion matrix percentages")
    option_method = st.selectbox("Method",df_methods_pie_chart["method"], index = 0)
    st.plotly_chart(pie_chart(r_pie_,option_method))