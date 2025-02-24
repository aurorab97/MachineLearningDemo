# X Fork
# --- IMPORT SECTION ---
import math
import pandas as pd # for DataFrames
import numpy as np # for numpy array operations
import matplotlib.pyplot as plt # for visualization
import seaborn as sns
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
x = data["YearsExperience"].values.reshape(-1,1)
y = data["Salary"].values.reshape(-1,1)
# using a plot to visualize the data
plt.title("Years of Experience vs Salary") # title of the plot
plt.xlabel("Years of Experience") # title of x axis
plt.ylabel("Salary") # title of y axis
plt.scatter(x, y, color="red") # actual plot
sns.regplot(data = data, x = "YearsExperience", y = "Salary") # regression line
plt.show() # renderize the plot to show it

# splitting the dataset into training and test (0.2 -- > 20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 101 )

# Checking the train and the test size
print(f"\nThe total x size is: {x.shape[0]}")
print(f"\nThe total x_train size is: {x_train.shape[0]}, and is the {x_train.shape[0] / x.shape[0] * 100}")
print(f"\nThe total x_test size is: {x_test.shape[0]}, and is the {x_test.shape[0] / x.shape[0] * 100}")

print(f"\nThe total y size is: {y.shape[0]}")
print(f"\nThe total y_train size is: {y_train.shape[0]}, and is the {y_train.shape[0] / y.shape[0] * 100}")
print(f"\nThe total y_test size is: {y_test.shape[0]}, and is the {y_test.shape[0] / y.shape[0] * 100}")

# visualizing data before scaling
print(f"\n-- BEFORE SCALING -- X_train:\n{x_train[:5]}")
print(f"\n-- BEFORE SCALING -- y_train:\n{y_train[:5]}")
print(f"\n-- BEFORE SCALING -- X_test:\n{x_test[:5]}")
print(f"\n-- BEFORE SCALING -- y_test:\n{y_test[:5]}")

# feature scaling
scaler = StandardScaler()
# we are going to scale ONLY the features (i.e. the X) and NOT the y!
x_train_scaled = scaler.fit_transform(x_train) # fitting to X_train and transforming them
x_test_scaled = scaler.transform(x_test) # transforming X_test. DO NOT FIT THEM!

# visualizing data after scaling
print(f"\n-- AFTER SCALING -- X_train:\n{x_train_scaled[:5]}")
print(f"\n-- AFTER SCALING -- y_train:\n{y_train[:5]}")
print(f"\n-- AFTER SCALING -- X_test:\n{x_test_scaled[:5]}")
print(f"\n-- AFTER SCALING -- y_test:\n{y_test[:5]}")

# linear regression
model = LinearRegression()
# performing the training on the train data (i.e x_train_scaled, x_train)
model.fit(x_train_scaled, y_train)
# predicting new values
y_pred = model.predict(x_test_scaled)
# visualizing the regression
print(f"\nAfter the training, the params for the Regressor are: {model.coef_}")

# visuazing the regression
plt.title("Year of Experience vs Salary")
plt.xlabel("Year of Experience")
plt.ylabel("Salary")
plt.scatter(x_test, y_test, color = 'red', label = 'Real Data')
plt.plot(x_test, y_pred, color = 'blue', label = 'Predicted Data')
plt.legend()
plt.show()

# evaluating the model
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nRMSE: {rmse:.2f}")

# --- END OF MAIN CODE ---
