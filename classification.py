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

# feature scaling
scaler = StandardScaler()
# we are going to scale ONLY the features (i.e. the X) and NOT the y!
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# creating model
model = RandomForestClassifier(n_estimators = 100, random_state = 101)

# training model
model.fit(x_train_scaled, y_train)

# prediction over the test set
y_pred = model.predict(x_test_scaled)

# evaluting the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nThe accuracy of the model is: {accuracy * 100:.2f}")

# classification report
print(f"\nClassification report:\n{classification_report(y_test, y_pred)}")

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = dataset.target_names,
            yticklabels = dataset.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

