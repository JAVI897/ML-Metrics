#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
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

  def PRC_Value(self):
    precision = []
    recall = []
    for i in np.linspace(1.0, 0.0, num=100):
      aux = Metrics(self.test, self.prediction, threshold = i)
      precision_value = aux.precision()
      sensitivity_value = aux.sensitivity()
      if precision_value != None: #Con un threshold de 1 no existen ni FP ni TP,
      # entonces precision: ZeroDivisionError-> salta el except y devuelve None

        precision.append(precision_value)
        recall.append(sensitivity_value)

    #precision, recall, _ = precision_recall_curve(self.test, self.prediction)
    return precision, recall

  def ROC_Value(self):
    recall = []
    false_positive_rate = []
    for i in np.linspace(1.0, 0.0, num=100):
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
  
  def plot_ROC(self, fill = False, legend = True, double = False, threshold = False,methods=None):
    if double is False:
      plt.style.use("ggplot")
      plt.figure(figsize = (9, 9))
    for i, (Y_test, prediction) in enumerate(self.data):
      metrics = Metrics(Y_test, prediction)
      recall, false_positive_rate = metrics.ROC_Value()
      AUC = metrics.auc()

      plt.plot(false_positive_rate, recall, label="ROC curve Model {} (AUC = {})".format(i + 1, AUC))

      if threshold: # Hay que revisarlo para que de los de youden o más o los que quiera el usuario
        opt = Optimum(Y_test, prediction)
        threshold_youden, object_max_youden = opt.optimum_by_youden()
        threshold_f_score, object_f_score = opt.optimum_by_f_score()
        threshold_ROC, object_ROC = opt.optimum_for_ROC()
        threshold_difference_S_S, object_difference_S_S = opt.optimum_by_sensitivity_specificity_difference()
        
        
        if "Youden" in methods:
            plt.scatter(object_max_youden.false_positive_rate(), object_max_youden.sensitivity(), marker='X',
                        label = "Youden threshold {} Model {}".format(threshold_youden, i+1))
        
        if "F-score" in methods:
            plt.scatter(object_f_score.false_positive_rate(), object_f_score.sensitivity(), marker='X',
                        label = "F-score threshold {} Model {}".format(threshold_f_score, i+1))
        
        if "Distance ROC" in methods:
            plt.scatter(object_ROC.false_positive_rate(), object_ROC.sensitivity(), marker='X',
                        label = "Distance_ROC threshold {} Model {}".format(threshold_ROC, i+1))
        
        if "Difference Sensitivity-Specificity" in methods:
            plt.scatter(object_difference_S_S.false_positive_rate(), object_difference_S_S.sensitivity(), marker='X',
                        label = "Sensitivity_Specificity_Difference threshold {} Model {}".format(threshold_difference_S_S, i+1))      
     
      if fill:
        plt.fill_between(false_positive_rate, recall, alpha=0.5)
        plt.plot(np.linspace(1.0, 0.0, num=100), np.linspace(1.0, 0.0, num=100), linestyle = "--", linewidth=1, color = "grey")
      else:
        plt.plot(np.linspace(1.0, 0.0, num=100), np.linspace(1.0, 0.0, num=100), linestyle = "--", linewidth=0.5, color = "grey")
    
    plt.title("ROC-curve")
    if fill is True and double is True:
      plt.title("")
    plt.xlabel("false_positive_rate")
    plt.ylabel("true_positive_rate")
    if legend:
      plt.legend(fontsize ='medium')
    plt.grid(False)
    #if double is False:
      #plt.show()

  def plot_PRC(self, fill = False, legend = True, double = False, threshold = False, methods=None):
    if double is False:
      plt.style.use("ggplot")
      plt.figure(figsize = (9, 9))
    for i, (Y_test, prediction) in enumerate(self.data):
      metrics = Metrics(Y_test, prediction)
      precision, recall = metrics.PRC_Value()
      AP = metrics.ap()
      plt.plot(recall, precision, label = "Precision-recall-curve Model {} (AP = {})".format(i+1, AP))
      
      if threshold: 
        opt = Optimum(Y_test, prediction)
        threshold_PRC, object_PRC = opt.optimum_for_PRC()
        threshold_difference_R_P, object_difference_R_P = opt.optimum_by_recall_precision_difference()
        
        if "Distance PRC" in methods:
            plt.scatter(object_PRC.sensitivity(), object_PRC.precision(), marker='X',
                        label = "Distance_PRC threshold {} Model {}".format(threshold_PRC, i+1))
        else:pass
        if "Difference Recall-Precision" in methods:
            plt.scatter(object_difference_R_P.sensitivity(), object_difference_R_P.precision(), marker='X',
                        label = "Difference_Recall_Precision threshold {} Model {}".format(threshold_difference_R_P, i+1))      
        else:pass

      if fill:
        plt.fill_between(recall, precision, alpha=0.5)
      plt.plot(np.linspace(0.0, 1.0, num=100), np.linspace(1.0, 0.0, num=100), linewidth=0.0)

    plt.title("Precision-recall-curve")
    if fill is True and double is True:
      plt.title("")
    plt.xlabel("Recall(sensitivity)")
    plt.ylabel("Precision(PPV)")
    if legend:
      plt.legend(fontsize ='medium')
    plt.grid(False)
    #if double is False:
      #plt.show()

  def plot_all(self, fill = True, legend = True, threshold = False,methods=None):
    plt.style.use("ggplot")
    plt.figure(figsize = (15, 20))
    plt.subplot(2,2,1)
    self.plot_PRC(double=True, threshold = threshold, legend = legend,methods=methods)
    plt.subplot(2,2,2)
    self.plot_ROC(double=True, threshold = threshold, legend = legend,methods=methods)
    if fill:
      plt.subplot(2,2,3)
      self.plot_PRC(fill = True, double=True, legend = legend)
      plt.subplot(2,2,4)
      self.plot_ROC(fill = True, double=True, legend = legend)
    #plt.show()
  
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