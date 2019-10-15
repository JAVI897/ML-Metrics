#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects  as go
import math
import streamlit as st
from plotly.subplots import make_subplots
import seaborn as sns

class Metrics:

  def __init__(self, test, prediction, threshold = 0.5):
    self.test = test
    self.prediction = prediction
    self.tabla = np.hstack([test, prediction])
    self.threshold = threshold
    self.TP = 0; self.FN = 0; self.FP = 0; self.TN = 0;
    for i in self.tabla:
      if i[0] == 0 and i[1] <= self.threshold:
        self.TN += 1
      elif i[0] == 1 and i[1] >= self.threshold:
        self.TP += 1
      elif i[0] == 1 and i[1] <= self.threshold:
        self.FN += 1
      elif i[0] == 0 and i[1] >= self.threshold:
        self.FP += 1

  def confusion_matrix(self):
    return np.array([[self.TP, self.FP], [self.FN, self.TN]])
    
  def specificity(self):
    try:
      return np.round(self.TN/(self.TN+self.FP), decimals = 3)

    except:
      return None

  def sensitivity(self):
    try:
      return np.round(self.TP/(self.TP+self.FN), decimals = 3)

    except:
      return None

  def false_positive_rate(self):
    try:
      return np.round(self.FP/(self.FP+self.TN), decimals = 3)

    except:
      return None

  def precision(self):
    try:
      return np.round(self.TP/(self.TP+self.FP), decimals = 3)

    except:
      return None

  def PRC_Value(self, number_threshold = 100):
    precision = []
    recall = []
    for i in np.linspace(1.0, 0.0, num=number_threshold):
      aux = Metrics(self.test, self.prediction, threshold = i)
      precision_value = aux.precision()
      sensitivity_value = aux.sensitivity()
      if precision_value != None: #Con un threshold de 1 no existen ni FP ni TP,
      # entonces precision: ZeroDivisionError-> salta el except y devuelve None

        precision.append(precision_value)
        recall.append(sensitivity_value)

    #precision, recall, _ = precision_recall_curve(self.test, self.prediction)
    return precision, recall

  def ROC_Value(self, number_threshold = 100):
    recall = []
    false_positive_rate = []
    for i in np.linspace(1.0, 0.0, num=number_threshold):
      aux = Metrics(self.test, self.prediction, threshold = i)
      recall.append(aux.sensitivity())
      false_positive_rate.append(aux.false_positive_rate())

    return recall, false_positive_rate

  def auc(self, trapeze=False): #Area-under-the-curve for ROC
    recall, false_positive_rate = self.ROC_Value()
    if trapeze is False:
      return (sum([recall[i] * (false_positive_rate[i+1] - false_positive_rate[i]) for i in range(0, len(recall)-1)])).round(3)
    else:
      return (np.trapz(recall,false_positive_rate)).round(3)
    #auc = roc_auc_score(self.test, self.prediction)
  
  def ap(self, trapeze=False): #Area-under-the-curve for PRC
    precision, recall = self.PRC_Value()
    if trapeze is False:
      return (sum([precision[i] * (recall[i+1] - recall[i]) for i in range(0, len(precision)-1)])).round(3)
    else:
      return (np.trapz(precision,recall)).round(3)

  def youden_index(self):
    return (self.sensitivity() + self.specificity() - 1).round(3) #the optimum cut-off point, height above the chance line

  def f1(self):
    try:
      precision = self.precision()
      sensitivity = self.sensitivity()
      return (2 *( (precision*sensitivity) / (precision + sensitivity) )).round(3)
    except:
      return None


class Graphs:

  def __init__(self, data):
    self.data = data #Lista que contiene tuplas de la forma: [(Y_test, prediction)]

  def plot_ROC_plotly(self, double = False, fill = False, legend = True, threshold = False, methods=None, number_threshold = 100):
    fig = go.Figure()
    for i, (Y_test, prediction) in enumerate(self.data):
        metrics = Metrics(Y_test, prediction)
        recall, false_positive_rate = metrics.ROC_Value(number_threshold)
        AUC = metrics.auc()
        linspace = list(np.linspace(0.0, 1.0, num=100))
        fig.add_trace(go.Scatter(x=linspace, y = linspace, mode = 'lines', showlegend = False, line=dict(width=0.5)))
        fig.add_trace(go.Scatter(x=false_positive_rate, y=recall, mode='lines', name="ROC curve Model {} (AUC = {})".format(i + 1, AUC)))
        if threshold:
            opt = Optimum(Y_test, prediction)
            
            if "Youden" in methods:
                threshold_youden, object_max_youden = opt.optimum_by_youden()
                fig.add_trace(go.Scatter(x=[object_max_youden.false_positive_rate()], y=[object_max_youden.sensitivity()], 
                                             name= "Youden threshold {} Model {}".format(threshold_youden, i+1),
                                             mode = "markers",
                                             line=dict(width=4, dash='dot')))
                
            if "F-score" in methods:
                threshold_f_score, object_f_score = opt.optimum_by_f_score()
                fig.add_trace(go.Scatter(x=[object_f_score.false_positive_rate()], y=[object_f_score.sensitivity()], 
                                             name= "F-score threshold {} Model {}".format(threshold_f_score, i+1),
                                             mode = "markers",
                                             line=dict(width=4, dash='dot')))
                
            if "Distance ROC" in methods:
                threshold_ROC, object_ROC = opt.optimum_for_ROC()
                fig.add_trace(go.Scatter(x=[object_ROC.false_positive_rate()], y=[object_ROC.sensitivity()], 
                                             name= "Distance_ROC threshold {} Model {}".format(threshold_ROC, i+1),
                                             mode = "markers",
                                             line=dict(width=4, dash='dot')))
                
            if "Difference Sensitivity-Specificity" in methods:
                threshold_difference_S_S, object_difference_S_S = opt.optimum_by_sensitivity_specificity_difference()
                fig.add_trace(go.Scatter(x=[object_difference_S_S.false_positive_rate()], y=[object_difference_S_S.sensitivity()], 
                                             name= "Sensitivity_Specificity_Difference threshold {} Model {}".format(threshold_difference_S_S, i+1),
                                             mode = "markers",
                                             line=dict(width=4, dash='dot')))
                
        if fill:
            fig.add_trace(go.Scatter(x=false_positive_rate, y=recall,
                    mode='lines',
                    name="ROC curve Model {} (AUC = {})".format(i+1, AUC),
                    fill = 'tozeroy'))
            
    if double == False:
        if fill:
            dicc = dict(x=-.1, y=1.08 + 0.03 * (len(methods) if methods != None else 0) + 0.05)
        else:
            dicc = dict(x=-.1, y=1.08 + 0.03 * (len(methods) if methods != None else 0))
        fig.update_layout(showlegend=legend, legend=dicc, autosize=False, 
                      width=702, height=900, xaxis_title="false_positive_rate", yaxis_title="true_positive_rate")
    return fig
            
  
  def plot_PRC_plotly(self, double = False, fill = False, legend = True, threshold = False, methods=None, number_threshold = 100):
    fig = go.Figure()
    for i, (Y_test, prediction) in enumerate(self.data):
        metrics = Metrics(Y_test, prediction)
        precision, recall = metrics.PRC_Value(number_threshold)
        AP = metrics.ap()
        linspace = list(np.linspace(0.0, 1.0, num=100))
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name="Precision-recall-curve Model {} (AP = {})".format(i+1, AP)))
        fig.add_trace(go.Scatter(x=linspace, y = linspace, mode = 'lines', showlegend = False, line=dict(width=0.0)))
        
        if threshold:
            opt = Optimum(Y_test, prediction)
            if "Distance PRC" in methods:
                threshold_PRC, object_PRC = opt.optimum_for_PRC()
                fig.add_trace(go.Scatter(x=[object_PRC.sensitivity()], y=[object_PRC.precision()], 
                                             name="Distance_PRC threshold {} Model {}".format(threshold_PRC, i+1),
                                             mode = "markers",
                                             line=dict(width=4, dash='dot')))
                
            if "Difference Recall-Precision" in methods:
                threshold_difference_R_P, object_difference_R_P = opt.optimum_by_recall_precision_difference()
                fig.add_trace(go.Scatter(x=[object_difference_R_P.sensitivity()], y=[object_difference_R_P.precision()], 
                                             name="Difference_Recall_Precision threshold {} Model {}".format(threshold_difference_R_P, i+1),
                                             mode = "markers",
                                             line=dict(width=4, dash='dot')))
            
        if fill:
            fig.add_trace(go.Scatter(x=recall, y=precision,
                    mode='lines',
                    name="Precision-recall-curve Model {} (AP = {})".format(i+1, AP),
                    fill = 'tozeroy'))
            
    if double == False:
        if fill:
            dicc = dict(x=-.1, y=1.08 + 0.03 * (len(methods) if methods != None else 0) + 0.05)
        else:
            dicc = dict(x=-.1, y=1.08 + 0.03 * (len(methods) if methods != None else 0))
        fig.update_layout(showlegend=legend, legend=dicc, autosize=False, 
                      width=702, height=900, xaxis_title='Recall(sensitivity)', yaxis_title='Precision(PPV)')
    return fig
        


  def plot_all_plotly(self, fill =  False, legend = True, threshold = False, methods=None, number_threshold = 100):
    fig1 = self.plot_PRC_plotly(double = True, fill = fill, legend = legend, threshold = threshold, methods=methods, number_threshold = number_threshold)
    fig2 = self.plot_ROC_plotly(double = True, fill = fill, legend = legend, threshold = threshold, methods=methods, number_threshold = number_threshold)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('PRC', 'ROC'))
    for j in fig1.data:
        fig.append_trace(j, 1, 1)
    for i in fig2.data:
        fig.append_trace(i, 1, 2)
        
    if fill:
        dicc = dict(x=-.1, y=1.25 + 0.05 * (len(methods) if methods != None else 0) + 0.1)
    else:
        dicc = dict(x=-.1, y=1.25 + 0.05 * (len(methods) if methods != None else 0))
    fig.layout.update(showlegend=legend, legend=dicc, autosize=False, height=670, width=770)
    fig.layout.xaxis1.update(title='Recall', showgrid=False)
    fig.layout.yaxis1.update(title='Precision(PPV)', showgrid=False)
    
    fig.layout.xaxis2.update(title='true_positive_rate', showgrid=False)
    fig.layout.yaxis2.update(title='false_positive_rate', showgrid=False)
    return fig



  def plot_precision_recall_vs_threshold_plotly(self, legend = True, threshold = False, methods=None, number_threshold = 100):
    fig = go.Figure()
    for i, (Y_test, prediction) in enumerate(self.data):
        
            metrics = Metrics(Y_test, prediction)
            precisions, recalls = metrics.PRC_Value(number_threshold)
            fig.add_trace(go.Scatter(x=np.linspace(1.0, 0.0, num=len(precisions)-1), y=precisions[:-1],
                    mode='lines',
                    name='Precision'))
            
            fig.add_trace(go.Scatter(x=np.linspace(1.0, 0.0, num=len(precisions)-1), y=recalls[:-1],
                    mode='lines',
                    name='Recall'))
            
            if threshold:
                opt = Optimum(Y_test, prediction)
                if "Distance PRC" in methods:
                    threshold_PRC, object_PRC = opt.optimum_for_PRC()
                    fig.add_trace(go.Scatter(x=[threshold_PRC], y=[object_PRC.precision()], 
                                             name="Distance_PRC threshold precision {} ".format(threshold_PRC),
                                             mode = "markers",
                                             line=dict(color='royalblue', width=4, dash='dot')))
                    
                    fig.add_trace(go.Scatter(x=[threshold_PRC], y=[object_PRC.sensitivity()],
                                             showlegend=False,
                                             line=dict(color='royalblue', width=4, dash='dot')))

                if "Difference Recall-Precision" in methods:
                    threshold_difference_R_P, object_difference_R_P = opt.optimum_by_recall_precision_difference()
                    fig.add_trace(go.Scatter(x=[threshold_difference_R_P], y=[object_difference_R_P.precision()], 
                                             name="Difference Recall-Precision threshold precision {} ".format(threshold_difference_R_P),
                                             mode = "markers",
                                             line=dict(color='firebrick', width=4, dash='dot')))
                    
                    fig.add_trace(go.Scatter(x=[threshold_difference_R_P], y=[object_difference_R_P.sensitivity()],
                                             showlegend=False,
                                             line=dict(color='firebrick', width=4, dash='dot')))
                    
            
    # Edit the layout
    dicc = dict(x=-.1, y=1.08 + 0.05 * (len(methods) if methods != None else 0))
    fig.update_layout(showlegend=legend, legend=dicc, autosize=False, width=702, height=900, xaxis_title='Decision threshold',yaxis_title='Score')
    return fig
            
  
class Optimum:

  def __init__(self, test, prediction):
    self.test = test
    self.prediction = prediction
  
  def optimum_by_youden(self):
    object_max = None
    threshold = None
    youden = 0
    for i in np.linspace(1.0, 0.0, num=100):
      aux = Metrics(self.test, self.prediction, threshold = i)
      youden_index = aux.youden_index()
      if youden_index > youden:
        youden = youden_index
        object_max = aux
        threshold = i
    return threshold.round(2), object_max

  def optimum_for_ROC(self):
    object_min = None
    threshold = None
    distance = math.sqrt(2)
    for i in np.linspace(1.0, 0.0, num=100):
      aux = Metrics(self.test, self.prediction, threshold = i)
      distance_aux = math.sqrt(
          (aux.false_positive_rate())**2 + (aux.sensitivity() - 1)**2
      )
      if distance_aux < distance:
        distance = distance_aux
        object_min = aux
        threshold = i
    return threshold.round(2), object_min

  def optimum_for_PRC(self):
    object_min = None
    threshold = None
    distance = math.sqrt(2)
    for i in np.linspace(1.0, 0.0, num=100):
      aux = Metrics(self.test, self.prediction, threshold = i)
      if aux.precision() is not None:
        distance_aux = math.sqrt(
            (aux.sensitivity() - 1)**2 + (aux.precision() - 1)**2
        )
        if distance_aux < distance:
          distance = distance_aux
          object_min = aux
          threshold = i
    return threshold.round(2), object_min

  def optimum_by_f_score(self):
    object_max = None
    threshold = None
    f_score = 0 # F1 score reaches its best value at 1 and worst at 0
    for i in np.linspace(1.0, 0.0, num=100):
      aux = Metrics(self.test, self.prediction, threshold = i)
      f_score_aux = aux.f1()
      if f_score_aux != None and f_score_aux > f_score:
        f_score = f_score_aux
        object_max = aux
        threshold = i
    return threshold.round(2), object_max

  
  def optimum_by_sensitivity_specificity_difference(self):
    object_min=None
    threshold=None
    dif_sensitivity_specificity=[]
    for i in np.linspace(1.0,0.0,num=100):
      aux=Metrics(self.test,self.prediction,threshold=i)
      sensitivity_aux= aux.sensitivity()
      specificity_aux= aux.specificity()
      dif = abs(sensitivity_aux - specificity_aux)
      dif_sensitivity_specificity.append(dif)
      if dif == min(dif_sensitivity_specificity):
        object_min= aux
        threshold= i
    return threshold.round(2),object_min
  
  
  def optimum_by_recall_precision_difference(self):
    object_min=None
    threshold=None
    dif_recall_precision=[]
    for i in np.linspace(1.0,0.0,num=100):
      aux=Metrics(self.test,self.prediction,threshold=i)
      recall_aux= aux.sensitivity()
      precision_aux= aux.precision()
      if precision_aux!= None:
        dif = abs(recall_aux - precision_aux)
        dif_recall_precision.append(dif)
        if dif == min(dif_recall_precision):
          object_min= aux
          threshold= i
    return threshold.round(2),object_min
  
  def report(self, colormap = True):
    threshold_youden, object_youden = self.optimum_by_youden()
    threshold_f_score, object_f_score = self.optimum_by_f_score()
    threshold_ROC, object_ROC = self.optimum_for_ROC()
    threshold_PRC, object_PRC = self.optimum_for_PRC()
    threshold_difference_S_S, object_difference_S_S = self.optimum_by_sensitivity_specificity_difference()
    threshold_difference_R_P, object_difference_R_P = self.optimum_by_recall_precision_difference()

    df = pd.DataFrame({"Threshold":[threshold_youden, threshold_f_score, threshold_ROC, threshold_PRC, threshold_difference_S_S,threshold_difference_R_P], 
                       "TP":[object_youden.TP, object_f_score.TP, object_ROC.TP, object_PRC.TP, object_difference_S_S.TP, object_difference_R_P.TP], 
                       "TN":[object_youden.TN, object_f_score.TN, object_ROC.TN, object_PRC.TN, object_difference_S_S.TN, object_difference_R_P.TN],
                       "FP":[object_youden.FP, object_f_score.FP, object_ROC.FP, object_PRC.FP, object_difference_S_S.FP, object_difference_R_P.FP], 
                       "FN":[object_youden.FN, object_f_score.FN, object_ROC.FN, object_PRC.FN, object_difference_S_S.FN, object_difference_R_P.FN],
                       "Specificity":[object_youden.specificity(), object_f_score.specificity(), object_ROC.specificity(),
                                      object_PRC.specificity(),object_difference_S_S.specificity(), object_difference_R_P.specificity()], 
                       "Sensitivity":[object_youden.sensitivity(), object_f_score.sensitivity(), object_ROC.sensitivity(),
                                      object_PRC.sensitivity(),object_difference_S_S.sensitivity(), object_difference_R_P.sensitivity()], 
                       "PPV":[object_youden.precision(), object_f_score.precision(), object_ROC.precision(),
                              object_PRC.precision(), object_difference_S_S.precision(), object_difference_R_P.precision()], 
                       "AUC":[object_youden.auc() for i in range(6)], 
                       "AP":[object_youden.ap() for i in range(6)]},
                      index = ["Youden", "F-score", "Distance_ROC", "Distance_PRC","Difference_Sensitivity_Specificity","Difference_Recall_Precision"])
    if colormap:
      cm = sns.light_palette("green", as_cmap=True)
      s = df.sort_values(by='Threshold',ascending=False).style\
             .background_gradient(cmap = cm, high = 0.5, low = -0.5, axis = 0)
             #.set_properties(**{'width': '75px', 'text-align': 'center', 'font-size': '10pt'})
      return s

    return df