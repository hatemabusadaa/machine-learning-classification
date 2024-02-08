from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("Celiac disease.csv")

# Drop unnecessary columns
data = data.drop(columns=['age', 'Timestamp'])

# Convert categorical data to numerical using LabelEncoder
categorical_features = data.drop(columns=['Medical diagnosis']).columns.tolist()
encoder = LabelEncoder()
data[categorical_features] = data[categorical_features].apply(encoder.fit_transform)

# Split the data into training and testing sets
x = data.drop(columns=['Medical diagnosis'])
y = data["Medical diagnosis"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Use the Multinomial Naive Bayes classifier
classifier = MultinomialNB()

# Fit the classifier on the training data
classifier.fit(x_train, y_train)

# Make predictions on the test data
y_probs = classifier.predict_proba(x_test)[:, 1]  # Use predict_proba for probabilities
y_pred = y_probs > 0.4  # Threshold probabilities for binary predictions

# Map 'no' and 'yes' labels to False and True
y_test_mapped = y_test.map({'no': False, 'yes': True})

# Calculate ROC AUC
roc_auc = roc_auc_score(y_test_mapped, y_probs)

# Print hold-out validation results
print('Hold-Out')
print(f"Mean AUC: {roc_auc:.2f}")
print('=========================================================================')

# Print confusion matrix, accuracy, and classification report
print('Confusion Matrix:')
print(confusion_matrix(y_test_mapped, y_pred))
print('=========================================================================')
accuracy = accuracy_score(y_test_mapped, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print('=========================================================================')
