from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

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

# Split the data into training and testing sets (Hold-out)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Preprocess the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the KNN model
k = 10
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy for hold-out
accuracy_hold_out = accuracy_score(y_test, y_pred)
print("Accuracy (Hold-out):", accuracy_hold_out)

# Calculate ROC curve and ROC AUC score
y_prob = knn.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

print('ROC AUC score(hold out)   ',roc_auc)

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

# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
