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
import plotly.graph_objects as go


prediction_1=np.load("prediction_1.npy")
Y_Test_1=np.load("Y_Test_1.npy")

data = [(Y_Test_1, prediction_1)]
graphs = metrics.Graphs(data)
optimum= metrics.Optimum(Y_Test_1,prediction_1)

prediction1_input = st.text_input("Add data", "Write your path:")
    
    
#st.write(prediction1_input) 
#if prediction1_input!= "Write your path:":
    #prediction_1= np.load(prediction1_input)
    #Y_Test_1=np.load("Y_Test_1.npy")
    
    #data = [(Y_Test_1, prediction_1)]
    #graphs = metrics.Graphs(data)
    #optimum= metrics.Optimum(Y_Test_1,prediction_1)




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

df_methods_pie_chart= pd.DataFrame({"method":["Youden","F-score","Distance_ROC",
                                              "Distance_PRC","Difference_Sensitivity_Specificity",
                                              "Difference_Recall_Precision"]})


#FUNCTIONS
    
def configuration(other_graph = False):
    threshold=False
    fill=False
    legend=True
    number_threshold = 100
    if st.sidebar.checkbox("Show settings"):
        #Threshold visualization
        option_threshold= st.sidebar.selectbox("Threshold",list(df_binary["threshold"]), index = 0)
        threshold = True if option_threshold == "Yes" else False

        #Fill visualization
        if other_graph == False:
            option_fill=  st.sidebar.selectbox("Fill",list(df_binary["fill"]), index = 0)
            fill = True if option_fill == "Yes" else False
        if other_graph:
            fill = None
        #Legend visualization
        option_legend= st.sidebar.selectbox("Legend",list(df_binary["legend"]), index = 0)
        legend = True if option_legend == "Yes" else False
        
        number_threshold = st.sidebar.slider("Number of thresholds:", min_value = 0, 
                                             max_value = 100, value = 100)
       
    return threshold,fill,legend, number_threshold

def configuration_report():
    colormap = True
    if st.sidebar.checkbox("Show settings"):
        #colormap
        option_colormap = st.sidebar.selectbox("Colormap",df_binary["colormap"])
        colormap = True if option_colormap == "Yes" else False
     
    return colormap

def grafico(option_curve, option_threshold,option_fill,option_legend,methods, number_threshold):
    
    if option_curve == "ROC Curve":
        g=graphs.plot_ROC(threshold= option_threshold, fill=option_fill,methods=methods,
                          legend=option_legend, number_threshold = number_threshold)
        return st.pyplot(g, transparent = False, optimize = True,
                              quality = 100, bbox_inches="tight")
        
    elif option_curve == "PRC Curve":
        g=graphs.plot_PRC(threshold= option_threshold, fill=option_fill, methods=methods,
                          legend=option_legend, number_threshold = number_threshold)
        return st.pyplot(g, transparent = False, optimize = True,
                              quality = 100, bbox_inches="tight")
        
    elif option_curve == "Both":
        g=graphs.plot_all(threshold= option_threshold, fill=option_fill,methods=methods,
                          legend=option_legend, number_threshold = number_threshold)        
        return st.pyplot(g, transparent = False, optimize = True,
                              quality = 100, bbox_inches="tight")

def other_graphics(option_graphs, option_threshold,option_legend,methods, number_threshold):
    
    if option_graphs == "Precision and Recall vs Decision threshold":
        g=graphs.plot_precision_recall_vs_threshold(legend = option_legend, threshold = option_threshold, 
                                                    methods=methods, number_threshold = number_threshold)
        return st.pyplot(g, transparent = False, optimize = True,
                              quality = 100, bbox_inches="tight")

def pie_chart(df,method):
    dic_values={}
    header_list=["TN","TP","FN","FP"]
    df=df[header_list]
    for index,value in df.iterrows():
        v = list(df.loc[index].values)
        dic_values[index]=v
    
    fig = go.Figure(data=[go.Pie(labels=header_list, values=dic_values[method], hole=.3)])
    return fig

def methods_prc():
    option_method=st.multiselect("Methods:",df_methods_prc["model"])
    return option_method 

def methods_roc():
    option_method=st.multiselect("Methods:",df_methods_roc["model"])
    return option_method 

def methods_both():
    option_method=st.multiselect("Methods:",df_methods_both["model"])
    return option_method 

def methods_Precision_and_Recall():
    option_method=st.multiselect("Methods:",df_methods_prc["model"])
    return option_method 


#INTERFACE 

st.sidebar.text("Options")
report= st.sidebar.checkbox("Report")
curve_type = st.sidebar.checkbox("Curve type")
other_graphs = st.sidebar.checkbox("Other graphs")

if report:
    st.sidebar.title("Report")
    report_sidebar= st.sidebar.checkbox("Show report")
    
    if report_sidebar:
        st.title("Report")
        colormap = configuration_report()
        r= optimum.report(colormap=colormap)
        st.dataframe(r)
        r_pie=optimum.report(colormap=False)
        
        st.title("Confusion matrix percentages")
        option_method= st.selectbox("Method",df_methods_pie_chart["method"])
        fig= pie_chart(r_pie,option_method)
        st.plotly_chart(fig) #da error si en report colormap == True
    
if curve_type:
    st.sidebar.title("Curve type")
    option_curve= st.sidebar.selectbox("Select",df_curves["curve"])
    option_threshold,option_fill,option_legend,number_threshold= configuration()
    
    
    if option_curve == "ROC Curve": 
    
        st.title("ROC Curve")
        if option_threshold: methods_list_roc= methods_roc() 
        else:methods_list_roc=None
        grafico(option_curve, option_threshold,option_fill,option_legend,methods_list_roc, number_threshold)
    
        
    elif option_curve == "PRC Curve": 
        
        st.title("PRC Curve")  
        if option_threshold: methods_list_prc= methods_prc()
        else: methods_list_prc=None
        grafico(option_curve, option_threshold,option_fill,option_legend,methods_list_prc, number_threshold)
             
    elif option_curve == "Both":
        st.title("ROC and PRC curves")
        if option_threshold: methods_list_both= methods_both()
        else:methods_list_both=None
        grafico(option_curve, option_threshold,option_fill,option_legend,methods_list_both, number_threshold)

if other_graphs:
    st.sidebar.title("Other graphs")
    option_graphs = st.sidebar.selectbox("Select",df_graphs["graphs"])
    option_threshold,option_fill,option_legend,number_threshold= configuration(other_graph = True)
    
    if option_graphs == "Precision and Recall vs Decision threshold":
        st.title("Precision and Recall vs Decision threshold")
        if option_threshold: 
            methods_list= methods_Precision_and_Recall()
        else: 
            methods_list=None
        other_graphics(option_graphs, option_threshold,option_legend,methods_list, number_threshold)
            
            

    