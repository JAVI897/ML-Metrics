"""Main module for the streamlit app"""
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
import plotly.graph_objects  as go
import src.st_extensions
import src.pages.home
import src.pages.report
import src.pages.curve_type
import src.pages.other_graphs

PAGES = {
    "Report": src.pages.report,
    "Curve type": src.pages.curve_type,
    "Other graphs": src.pages.other_graphs,
	"Home": src.pages.home
}

selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
src.st_extensions.write_page(page)