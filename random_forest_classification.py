# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('gameData.csv')  # Ensure the dataset contains the required columns

# Features
X = dataset[['lives', 'fakes', 'looker', 'doubleDamage', 'shield', 'heals', 'doubleTurn']].values

# Targets
y = dataset[['usedHeals', 'usedDoubleTurn', 'usedLooker', 'usedDoubleDamage', 'usedShield', 'shotHimself']].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

# Training the Random Forest Regression model on the Training set
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# Initialize the base model
base_model = RandomForestRegressor(n_estimators=10, criterion='mse', random_state=0)

# Create the multi-output regressor
multi_output_model = MultiOutputRegressor(base_model, n_jobs=-1)
multi_output_model.fit(X_train, y_train)

# Predicting a new result
new_data = [[3, 2, 1, 1, 1, 2, 1]]  # Example input
new_data_scaled = sc.transform(new_data)
predictions = multi_output_model.predict(new_data_scaled)
print(predictions)

# Predicting the Test set results
y_pred = multi_output_model.predict(X_test)
print(np.concatenate((y_pred, y_test), axis=1))

# Evaluating the model
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print('Mean Squared Error for each target:', mse)

# Visualising the results
# Note: Visualization for multi-output predictions can be complex. 
# Here, we can visualize predictions vs actuals for each target separately

targets = ['usedHeals', 'usedDoubleTurn', 'usedLooker', 'usedDoubleDamage', 'usedShield', 'shotHimself']

for i in range(len(targets)):
    plt.figure()
    plt.scatter(range(len(y_test)), y_test[:, i], color='red', label='Actual')
    plt.scatter(range(len(y_test)), y_pred[:, i], color='blue', label='Predicted')
    plt.title(f'{targets[i]} Predictions vs Actuals')
    plt.xlabel('Sample Index')
    plt.ylabel(targets[i])
    plt.legend()
    plt.show()
