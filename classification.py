# --- IMPORT SECTION ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.datasets import load_iris

# --- MAIN CODE ---
# importing the dataset: the Iris dataset contains data of three species of flower
dataset = load_iris()

# creating the dataset
data = pd.DataFrame(data = dataset.data, columns = dataset.feature_names)
data['target'] = dataset.target

# visualizing the first rows of the dataset
print(f"\nHere are the first 5 rows of dataset:\n{data.head()}")

# separate the data in features and target
x = data.iloc[:, :-1].values
y = data['target'].values

# splitting the dataset into trading
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 101 )