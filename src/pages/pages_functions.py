import streamlit as st
import plotly.graph_objects  as go
import pandas as pd

#REPORT FUNCTIONS

def configuration_report():
    
    colormap = True
    if st.sidebar.checkbox("Show settings"):
        #colormap
        option_colormap = st.sidebar.selectbox("Colormap",["Yes", "No"])
        colormap = True if option_colormap == "Yes" else False
     
    return colormap

def pie_chart(df,method):
    dic_values={}
    header_list=["TN","TP","FN","FP"]
    df=df[header_list]
    for index,value in df.iterrows():
        v = list(df.loc[index].values)
        dic_values[index]=v
    values = dic_values[method]
    fig = go.Figure(data=[go.Pie(labels=header_list, values=values, hole=.3)])
    return fig


#CURVE TYPE FUNCTIONS

def configuration(other_graph = False):
    
    df_binary= pd.DataFrame({"threshold":["No","Yes"],
                  "fill":["No","Yes"],
                  "legend":["Yes","No"]})
    
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

def methods_prc():
    df_methods_prc= pd.DataFrame({"model":["Distance PRC","Difference Recall-Precision"]})
    option_method=st.multiselect("Methods:",df_methods_prc["model"])
    return option_method 

def methods_roc():
    df_methods_roc= pd.DataFrame({"model":["Youden","F-score", "Distance ROC","Difference Sensitivity-Specificity"]})
    option_method=st.multiselect("Methods:",df_methods_roc["model"])
    return option_method 

def methods_both():
    df_methods_both= pd.DataFrame({"model":["Distance PRC","Difference Recall-Precision",
                                    "Youden","F-score", "Distance ROC","Difference Sensitivity-Specificity"]})
    option_method=st.multiselect("Methods:",df_methods_both["model"])
    return option_method

def grafico(graphs, option_curve, option_threshold,option_fill,option_legend,methods, number_threshold):
    
    if option_curve == "ROC Curve":
        g=graphs.plot_ROC(threshold= option_threshold, fill=option_fill,methods=methods,
                          legend=option_legend, number_threshold = number_threshold)
        return st.pyplot(g, transparent = False, optimize = True,
                              quality = 100, bbox_inches="tight")
        
    elif option_curve == "PRC Curve":
        g=graphs.plot_PRC_plotly(threshold= option_threshold, fill=option_fill, methods=methods,
                          legend=option_legend, number_threshold = number_threshold)
        return g
        
    elif option_curve == "Both":
        g=graphs.plot_all(threshold= option_threshold, fill=option_fill,methods=methods,
                          legend=option_legend, number_threshold = number_threshold)        
        return st.pyplot(g, transparent = False, optimize = True,
                              quality = 100, bbox_inches="tight")
    

#OTHER GRAPHS FUNCTIONS

def methods_Precision_and_Recall():
    df_methods_prc= pd.DataFrame({"model":["Distance PRC","Difference Recall-Precision"]})
    option_method=st.multiselect("Methods:",df_methods_prc["model"])
    return option_method

def other_graphics(graphs, option_graphs, option_threshold,option_legend,methods, number_threshold):
    
    if option_graphs == "Precision and Recall vs Decision threshold":
        return graphs.plot_precision_recall_vs_threshold_plotly(legend = option_legend, threshold = option_threshold, 
                                                    methods=methods, number_threshold = number_threshold)