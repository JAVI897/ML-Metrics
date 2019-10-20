"""Report page"""
import streamlit as st
import pandas as pd
import metrics
import numpy as np
import src.st_extensions
import plotly.graph_objects  as go
from src.pages.pages_functions import configuration_report, pie_chart
from src.pages.home import data

def write():
    """Method used to write page in app.py"""
    
    if len(data) == 0:
        st.error("There's not data")
        
    else:
        #Habría que cambiarlo de momento añadimos datos por aquí
        prediction_1=data[0][1]
        Y_Test_1=data[0][0]


        df_methods_pie_chart= pd.DataFrame({"method":["Youden","F-score","Distance_ROC",
                                                  "Distance_PRC","Difference_Sensitivity_Specificity",
                                                  "Difference_Recall_Precision"]})

        st.title("Report")
        st.sidebar.title("Report")
        colormap = configuration_report()

        @st.cache(ignore_hash=True)
        def report(colormap):
            optimum= metrics.Optimum(Y_Test_1,prediction_1)
            r= optimum.report(colormap=colormap)
            r_pie_= optimum.report(colormap=False)
            return r, r_pie_

        r, r_pie_ = report(colormap)
        st.dataframe(r)

        st.title("Confusion matrix percentages")
        option_method = st.selectbox("Method",df_methods_pie_chart["method"], index = 0)
        st.plotly_chart(pie_chart(r_pie_,option_method))