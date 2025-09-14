# Machine Learning Process
# Loading data
# Preprocessing
# Training a model
# Evaluating the model
# Making predictions

# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
# Loading the Dataset and Displaying the First Few Rows
iris_data = pd.read_csv('iris.csv')
iris_data.head()

# Sample output
#    Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm     Species
# 0   1            5.1           3.5            1.4           0.2 Iris-setosa
# 1   2            4.9           3.0            1.4           0.2 Iris-setosa
# 2   3            4.7           3.2            1.3           0.2 Iris-setosa
# 3   4            4.6           3.1            1.3           0.2 Iris-setosa
# 4   5            5.0           3.3            1.4           0.2 Iris-setosa

# Split the data into features (X) and labels (y)
X = iris_data.drop(columns=['Id', 'Species'])
y = iris_data['Species']

X.head()

# Sample output
#    SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm
# 0            5.1           3.5            1.4           0.2
# 1            4.9           3.0            1.4           0.2
# 2            4.7           3.2            1.3           0.2
# 3            4.6           3.1            1.3           0.2
# 4            5.0           3.3            1.4           0.2

# Create a ML model
model = LogisticRegression()

# Train the model
model.fit(X.values, y)

# Predict using the trained model
predictions = model.predict([[4.6, 3.5, 1.5, 0.2]])

# Print the predictions
print(predictions)
# 'Iris-setosa'



