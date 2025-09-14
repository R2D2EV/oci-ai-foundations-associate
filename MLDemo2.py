# Import necessary libraries
import pandas as pd  # Used for handling and analyzing structured data
import numpy as np  # Provides support for large, multi-dimensional arrays and matrices
from sklearn.linear_model import LogisticRegression  # Implements logistic regression for classification tasks
from sklearn.model_selection import train_test_split  # Splits data into training and testing sets
from sklearn.preprocessing import StandardScaler  # Standardizes features by removing the mean and scaling to unit variance
from sklearn.metrics import accuracy_score  # Measures the accuracy of classification models

# Load the dataset and display the first few rows
iris_data = pd.read_csv('iris.csv')
iris_data.head()

# Sample data:
# Id  SepalLengthCm   SepalWidthCm    PetalLengthCm   PetalWidthCm    Species
# 0   1               5.1             3.5             1.4             0.2         Iris-setosa
# 1   2               4.9             3.0             1.4             0.2         Iris-setosa
# 2   3               4.7             3.2             1.3             0.2         Iris-setosa
# 3   4               4.6             3.1             1.5             .02         Iris-setosa
# 4   5               .45              .34            .14              .23        Iris-setosa

# Split the data into features (X) and labels (y)
X = iris_data.drop(columns=['Id', 'Species'])
y = iris_data['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a ML model 
model = LogisticRegression()

# Train the model 
model.fit(X_train_scaled, y_train)
# Evaluate the model on the testing set
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Output:
# Accuracy: 1.0

# Sample new data for prediction
new_data = np.array([[5.1, 3.5, 1.4, 0.2],
                     [6.3, 2.9, 5.6, 1.8],
                     [4.9, 3.0, 1.4, 0.2]])

# Standardize the new data
new_data_scaled = scaler.transform(new_data)

# Make predictions
predictions = model.predict(new_data_scaled)

# Display the predicted classes
print("Predictions:", predictions)

# Output:
# Predictions: ['Iris-setosa' 'Iris-virginica' 'Iris-setosa']
