import pandas as pd # useful for loading the dataset
import numpy as np  # to perform array operations

# Load Dataset
dataset = pd.read_csv('path_to_your_dataset/ad_dataset.csv')

# Summarize Dataset
print(dataset.shape)
print(dataset.head(5))

# Segregate Dataset into X (Input/Independent Variable) & Y (Output/Dependent Variable)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Splitting Dataset into Train & Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

# Predicting whether a new customer with Age & Salary will Buy or Not
age = int(input("Enter New Customer Age: "))
sal = int(input("Enter New Customer Salary: "))
newCust = [[age, sal]]
result = model.predict(sc.transform(newCust))
print(result)
if result == 1:
    print("Customer will Buy")
else:
    print("Customer won't Buy")

# Prediction for all Test Data
y_pred = model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)

print("Accuracy of the Model: {0}%".format(accuracy_score(y_test, y_pred)*100))
