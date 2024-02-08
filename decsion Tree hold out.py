import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
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

# Convert target column to numerical using LabelEncoder
data['Medical diagnosis'] = encoder.fit_transform(data['Medical diagnosis'])


# Step 3: Data Preprocessing
X = data.drop('Medical diagnosis', axis=1)  # Replace 'target_column_name' with the actual target column
y = data['Medical diagnosis']



# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  # Adjust test_size as needed

# Step 5: Model Training
clf = DecisionTreeClassifier(random_state=42)  # You can adjust hyperparameters here
clf.fit(X_train, y_train)

# Step 6: Prediction and Accuracy Calculation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print()
print("Holdout")
print("Accuracy:", accuracy)

# Compute and print the Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate and print the AUC
auc = roc_auc_score(y_test, y_pred)  # Assuming y_pred are predicted class labels
print("AUC:", auc)
