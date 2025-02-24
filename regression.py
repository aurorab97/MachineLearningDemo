# X Fork
# --- IMPORT SECTION ---
import pandas as pd # for DataFrames
import numpy as np # for numpy array operations
import matplotlib.pyplot as plt # for visualization
# import seaborn as sns
# importing all the needed functions from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
# --- END OF IMPORT SECTION ---

# --- MAIN CODE ---
path_to_data = "Salary_Data.csv"
data = pd.read_csv(path_to_data)

# Visualizing the dataset
print(f"\nHere are the first 5 rows of the dataset:\n{data.head()}")

# separate the data in features and target
x = data["YearsExperience"]
y = data["Salary"]
# Using a plot to visualize the data
plt.title("Years of Experience vs Salary") # title of the plot
plt.xlabel("Years of Experience") # title of x axis
plt.ylabel("Salary") # title of y axis
plt.scatter(x, y, color="red") # actual plot
# sns.regplot(data= data, x="YearExperience", y="Salary")
plt.show() # renderize the plot to show it

# splitting the dataset into training and test (0.2 -- > 20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0 )

# Checking the train and the test size
print(f"\nThe total x size is: {x.shape[0]}")
print(f"\nThe total x_train size is: {x_train.shape[0]}")
print(f"\nThe total x_test size is: {x_test.shape[0]}")
# --- END OF MAIN CODE ---
