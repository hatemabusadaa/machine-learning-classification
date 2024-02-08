from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

data = pd.read_csv("Celiac disease.csv")

# Drop unnecessary columns
data = data.drop(columns=['age', 'Timestamp'])

# Convert categorical data to numerical using LabelEncoder
categorical_features = data.drop(columns=['Medical diagnosis']).columns.tolist()
encoder = LabelEncoder()
data[categorical_features] = data[categorical_features].apply(encoder.fit_transform)

# Split the data
x = data.drop(columns=['Medical diagnosis'])
y = data["Medical diagnosis"]

# Use the Multinomial Naive Bayes classifier
classifier = MultinomialNB()

# Create a cross-validation object with 10 folds
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation
y_probs = cross_val_predict(classifier, x, y, cv=cv, method='predict_proba')[:, 1]  # Use predict_proba for probabilities
y_encoded = encoder.transform(y)
roc_auc = roc_auc_score(y_encoded, y_probs)

# Print cross-validation results
print('Cross Validation')
print(f"Mean AUC: {roc_auc:.2f}")
print('=========================================================================')

# Print confusion matrix, accuracy, and classification report
y_predict = y_probs > 0.4  # Threshold probabilities for binary predictions
print('Confusion Matrix:')
print(confusion_matrix(y_encoded, y_predict))
print('=========================================================================')
accuracy = sum(y_encoded == y_predict) / len(y_encoded)
print(f"Accuracy: {accuracy:.2f}")
print('=========================================================================')
