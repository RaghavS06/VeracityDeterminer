#Import ALL the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import glob
import scikitplot as skplt
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV


#Load all the data
#****ADD FILE_PATH FOR FAKE DATASET****
fake_file_path = ""
fake_data = pd.read_csv(fake_file_path)

#****ADD FILE_PATH FOR TRUE DATASET****
true_file_path = "/content/True.csv"
true_data = pd.read_csv(true_file_path)

true_data = true_data.drop(['title', 'subject', 'date'], axis = 1)
true_data['veracity'] = pd.Series([1 for x in range(len(true_data.index))])

fake_data = fake_data.drop(['title', 'subject', 'date'], axis = 1)
fake_data['veracity'] = pd.Series([0 for x in range(len(fake_data.index))])

data = pd.concat([true_data, fake_data])
