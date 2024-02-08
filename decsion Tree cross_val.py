import pandas as pd
from sklearn.model_selection import cross_val_predict, train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
# Step 1: Load and Prepare Data


data = pd.read_csv("Celiac disease.csv")
print(data.head())
print(data.isnull().sum())
print('=========================================================================')

# Preprocessing
data = data.drop(columns=(['age', 'Timestamp']))
data = data.iloc[1:]
print(data.isnull().sum())
categorical_features = ['gender', 'relatives_diagnosed', 'type 1 diabetes', 'anemia', 'unwanted weight loss',
                      'bloating/gas', 'abdominal pain', 'vomiting /nausea', 'diarrhea', 'Frequent headaches', 'weak bones',
                      'lactose intolerance', 'itchy, blistery skin rash', 'fatigue/stress', 'constipation']
encoder = LabelEncoder()
data[categorical_features] = data[categorical_features].apply(encoder.fit_transform)
print(data.head())




X = data.drop("Medical diagnosis", axis=1)
y = data["Medical diagnosis"]

# Step 2: Splitting Data and Creating Decision Tree Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Step 3: Cross-Validation and Accuracy Calculation
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
mean_accuracy = scores.mean()
print()
print("Cross Validation")
print("Mean Cross-Validation Accuracy:", mean_accuracy)

# Step 4: Train and Evaluate on Test Set
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
#print("Test Set Accuracy:", test_accuracy)

# Step 5: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Step 6: Calculate AUC
y_prob = clf.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)
print("AUC Score:", auc_score)