"""Report page"""
import streamlit as st
import pandas as pd
import metrics
import numpy as np
import matplotlib.pyplot as plt
import src.st_extensions
import plotly.graph_objects  as go
from src.pages.pages_functions import configuration_report, pie_chart, nested_pie_chart
from src.pages.home import data

def write():
    """Method used to write page in app.py"""
    if len(data) == 0:
        st.error("There's not data")
        
    else:
        #Habría que cambiarlo de momento añadimos datos por aquí
        keys = list(data.keys())
        model = st.selectbox(label="Select a model:", options = keys, index = 0 )
        
        prediction=data[model][1]
        ground_truth=data[model][0]


        df_methods_pie_chart= pd.DataFrame({"method":["Youden","F-score","Distance_ROC",
                                                  "Distance_PRC","Difference_Sensitivity_Specificity",
                                                  "Difference_Recall_Precision"]})

        st.title("Report")
        st.sidebar.title("Report")
        colormap = configuration_report()

        @st.cache(allow_output_mutation=True)
        def report(colormap):
            optimum= metrics.Optimum(ground_truth,prediction)
            r= optimum.report(colormap=colormap)
            r_pie_= optimum.report(colormap=False)
            return r, r_pie_

        r, r_pie_ = report(colormap)
        st.dataframe(r)

        st.title("Confusion matrix percentages")
        option_method = st.selectbox("Method",df_methods_pie_chart["method"], index = 0)
        st.plotly_chart(pie_chart(r_pie_,option_method))
        
        st.title("Multi-level pie chart")
        labels = list(r_pie_.index)
        option_method2 = st.multiselect("Methods", options = labels, default = labels)
        st.pyplot(nested_pie_chart(r_pie_,option_method2))