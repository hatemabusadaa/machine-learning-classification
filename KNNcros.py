from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# read boyh
data=pd.read_csv("Celiac disease.csv")

# drop these tow f** columns
data=data.drop(columns=(['age','Timestamp']))

#wip first column out. why? no idea
data=data.iloc[1:]

catorigical_features=['gender','relatives_diagnosed','type 1 diabetes','anemia','unwanted weight loss',
                      'bloating/gas','abdominal pain','vomiting /nausea','diarrhea','Frequent headaches','weak bones',
                      'lactose intolerance','itchy, blistery skin rash','fatigue/stress','constipation']

encoder=LabelEncoder()
data[catorigical_features] = data[catorigical_features].apply(encoder.fit_transform)

# Load and prepare dataset
X, y = data[list(catorigical_features)].values, data['Medical diagnosis'].values
y = encoder.fit_transform(y)

# Create and train the KNN model
k = 10
knn = KNeighborsClassifier(n_neighbors=k)


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold cross-validation
y_probs = []
y_tests = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    knn.fit(X_train, y_train)
    
    y_probs.append(knn.predict_proba(X_test)[:, 1])
    y_tests.append(y_test)
    
y_probs = np.concatenate(y_probs)
y_tests = np.concatenate(y_tests)

fpr, tpr, thresholds = roc_curve(y_tests, y_probs)
roc_auc = roc_auc_score(y_tests, y_probs)

print('ROC AUC score(cross validation)   ',roc_auc)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

y_pred = (y_probs > 0.5).astype(int)  # Convert probabilities to binary predictions
conf_matrix = confusion_matrix(y_tests, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# Perform 10-fold cross-validation
cross_val_scores = cross_val_score(knn, X, y, cv=10)
mean_accuracy_cv = cross_val_scores.mean()
print("Mean Accuracy (10-Fold Cross-Validation):", mean_accuracy_cv)

