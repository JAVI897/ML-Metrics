#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 19:59:48 2019

@author: aidavillalba
"""
magicEnabled=False
import metrics 
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
import pandas as pd
import math
import seaborn as sns
import csv


prediction_1=np.load("prediction_1.npy")
Y_Test_1=np.load("Y_Test_1.npy")

data = [(Y_Test_1, prediction_1)]
graphs = metrics.Graphs(data)
optimum= metrics.Optimum(Y_Test_1,prediction_1)


#DATAFRAMES
df_binary= pd.DataFrame({"threshold":["No","Yes"],
                  "fill":["No","Yes"],
                  "legend":["Yes","No"],
                  "colormap":["Yes", "No"]})

df_curves=pd.DataFrame({"curve":["ROC Curve","PRC Curve","Both"]})
df_graphs=pd.DataFrame({"graphs":["Precision and Recall vs Decision threshold"]})

df_methods_prc= pd.DataFrame({"model":["Distance PRC","Difference Recall-Precision"]})
df_methods_roc= pd.DataFrame({"model":["Youden","F-score", "Distance ROC","Difference Sensitivity-Specificity"]})
df_methods_both= pd.DataFrame({"model":["Distance PRC","Difference Recall-Precision",
                                        "Youden","F-score", "Distance ROC","Difference Sensitivity-Specificity"]})
#FUNCTIONS
def configuration():
    threshold=False
    fill=False
    legend=True
    if st.sidebar.checkbox("Show settings"):
        #Threshold visualization
        option_threshold= st.sidebar.selectbox("Threshold",list(df_binary["threshold"]), index = 0)
        threshold = True if option_threshold == "Yes" else False

        #Fill visualization
        option_fill=  st.sidebar.selectbox("Fill",list(df_binary["fill"]), index = 0)
        fill = True if option_fill == "Yes" else False
        
        #Legend visualization
        option_legend= st.sidebar.selectbox("Legend",list(df_binary["legend"]), index = 0)
        legend = True if option_legend == "Yes" else False
       
    return threshold,fill,legend

def configuration_report():
    
    if st.sidebar.checkbox("Show settings"):
        #colormap
        option_colormap = st.sidebar.selectbox("Colormap",df_binary["colormap"])
        colormap = True if option_colormap == "Yes" else False
     
    return colormap

def grafico(option_threshold,option_fill,option_legend,methods):
    if option_curve == "ROC Curve":
        g=graphs.plot_ROC(threshold= option_threshold, fill=option_fill,methods=methods,legend=option_legend)
        return st.pyplot(g)
        
    elif option_curve == "PRC Curve":
        g=graphs.plot_PRC(threshold= option_threshold, fill=option_fill, methods=methods,legend=option_legend)
        return st.pyplot(g)
        
    elif option_curve == "Both":
        g=graphs.plot_all(threshold= option_threshold, fill=option_fill,methods=methods,legend=option_legend)        
        return st.pyplot(g, transparent = True, optimize = True,
                              quality = 100, bbox_inches="tight")

def methods_prc():
    option_method=st.multiselect("Methods:",df_methods_prc["model"])
    return option_method 

def methods_roc():
    option_method=st.multiselect("Methods:",df_methods_roc["model"])
    return option_method 

def methods_both():
    option_method=st.multiselect("Methods:",df_methods_both["model"])
    return option_method 


#INTERFACE 
st.sidebar.title("Report")
report_sidebar= st.sidebar.checkbox("Show report")

if report_sidebar:
    st.title("Report")
    colormap = configuration_report()
    r= optimum.report(colormap=colormap)
    st.dataframe(r) 
    
if not report_sidebar:
    st.sidebar.title("Curve type")
    option_curve= st.sidebar.selectbox("Select",df_curves["curve"])
    option_threshold,option_fill,option_legend= configuration()
    
    
    if option_curve == "ROC Curve": 
    
        st.title("ROC Curve")
        if option_threshold== True: methods_list_roc= methods_roc() 
        else:methods_list_roc=None
        grafico(option_threshold,option_fill,option_legend,methods_list_roc)
    
        
    elif option_curve == "PRC Curve": 
        
        st.title("PRC Curve")  
        if option_threshold== True: methods_list_prc= methods_prc()
        else: methods_list_prc=None
        grafico(option_threshold,option_fill,option_legend,methods_list_prc)
             
    elif option_curve == "Both":
        st.title("ROC and PRC curves")
        if option_threshold== True: methods_list_both= methods_both()
        else:methods_list_both=None
        grafico(option_threshold,option_fill,option_legend,methods_list_both)
    
    st.sidebar.title("Other graphs")
    option_graphs = st.sidebar.selectbox("Select",df_graphs["graphs"])
    