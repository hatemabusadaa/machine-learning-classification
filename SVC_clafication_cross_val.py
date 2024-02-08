from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

data = pd.read_csv("Celiac disease.csv")

# Drop unnecessary columns
#the age coulmn have alot of null values so we chooses to get rid of it
data=data.drop(columns=(['age','Timestamp']))
data=data.iloc[1:]

value_counts = data['Medical diagnosis'].value_counts()
# Create a pie chart
plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')

# Adding title
plt.title('Pie Chart of Category Distribution')
# Convert categorical data to numerical using LabelEncoder
categorical_features = data.drop(columns=['Medical diagnosis']).columns.tolist()
encoder = LabelEncoder()
data[categorical_features] = data[categorical_features].apply(encoder.fit_transform)

# Split the data
x = data.drop(columns=['Medical diagnosis'])
y = data["Medical diagnosis"]

# Standardize features by removing mean and scaling to unit variance
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Use the SVM classifier to fit data
classifier = SVC(kernel='linear')
num_features_to_select = 12
rfe_selector = RFE(estimator=classifier, n_features_to_select=num_features_to_select)

# Create a cross-validation object with 10 folds
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation
y_probs = cross_val_predict(rfe_selector, x, y, cv=cv, method='decision_function')
y_encoded = encoder.transform(y)
roc_auc = roc_auc_score(y_encoded, y_probs)


# Print cross-validation results
print ('cross validation')
print(f"Mean AUC: {roc_auc:.2f}")
print('=========================================================================')

# Print confusion matrix, accuracy, and classification report
y_predict = y_probs > 0  # Convert decision function values to binary predictions
print('Confusion Matrix:')
print(confusion_matrix(y_encoded, y_predict))
print('=========================================================================')
accuracy = sum(y_encoded == y_predict) / len(y_encoded)
print(f"Accuracy: {accuracy:.2f}")
print('=========================================================================')
