# AI_assignment
The thought process: Initially by using discription I have check the null values and data type of training data.
There is no null values in the data set. 
The assignment data has understood by using Exploratory Data Analysis.
EDA: I have use libraries for visualization. 1. seaborn 2. matplotlib
I have check the Outlires by using box plot.
Checked the data structure of target column by using pie chart.
Checked the correlation by using heatmap.
Checked the distribution of indivisual data by using plots.
Used RandomForest for model evalution and its hyperparameters.
Took test on test data.
Created new csv file for test data with "Prediction".
Created pickle file.
The Libraries used for the assignment are as follows:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import sklearn
import pickle
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import PowerTransforme
import pickle
