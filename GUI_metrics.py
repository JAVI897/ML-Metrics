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


prediction_1=np.load("prediction_1.npy")
Y_Test_1=np.load("Y_Test_1.npy")

data = [(Y_Test_1, prediction_1)]
graphs = metrics.Graphs(data)
optimum= metrics.Optimum(Y_Test_1,prediction_1)


#DATAFRAMES
df_binary= pd.DataFrame({"threshold":["No","Yes"],
                  "fill":["No","Yes"],
                  "legend":["Yes","No"]})

df_curves=pd.DataFrame({"curve":["ROC Curve","PRC Curve","Both"]})

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
        option_threshold= st.sidebar.selectbox("Threshold",df_binary["threshold"])
        if option_threshold == "Yes": 
            threshold= True

        #Fill visualization
        option_fill=  st.sidebar.selectbox("Fill",df_binary["fill"])
        if option_fill == "Yes": 
            fill= True
        #Legend visualization
        option_legend= st.sidebar.selectbox("Legend",df_binary["legend"])
        if option_legend == "No": 
            legend= False        
    return threshold,fill,legend

def grafico(option_threshold,option_fill,option_legend,methods):
    if option_curve == "ROC Curve":
        g=graphs.plot_ROC(threshold= option_threshold, fill=option_fill,methods=methods,legend=option_legend)
        return st.pyplot(g)
        
    elif option_curve == "PRC Curve":
        g=graphs.plot_PRC(threshold= option_threshold, fill=option_fill, methods=methods,legend=option_legend)
        return st.pyplot(g)
        
    elif option_curve == "Both":
        g=graphs.plot_all(threshold= option_threshold, fill=option_fill,methods=methods,legend=option_legend)        
        return st.pyplot(g)

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
        

else:
    st.title("Report")
    r= optimum.report()
    st.dataframe(r)    
    