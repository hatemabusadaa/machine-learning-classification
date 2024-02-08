#الحل النهائي لل cross validation #

from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

# تحميل البيانات وتنظيفها
# ...
data=pd.read_csv(r"Celiac disease.csv")
print(data.head())
print(data.isnull().sum())
print('=========================================================================')

#by looking at the dat we notice that the fiarst row was filled with nul vilues\
#the age coulmn have alot of null values so we chooses to get reed of them
data=data.drop(columns=(['age','Timestamp']))
data=data.iloc[1:]
#lets check now
print(data.isnull().sum())
catorigical_features=['gender','relatives_diagnosed','type 1 diabetes','anemia','unwanted weight loss',
                      'bloating/gas','abdominal pain','vomiting /nausea','diarrhea','Frequent headaches','weak bones',
                      'lactose intolerance','itchy, blistery skin rash','fatigue/stress','constipation']
encoder=LabelEncoder()
data[catorigical_features] = data[catorigical_features].apply(encoder.fit_transform)
print(data.head())



# إعداد X و y
# ...
X = data.drop(columns=['Medical diagnosis'])
y = data['Medical diagnosis']

# إعداد النموذج الفردي (مثلاً Decision Tree Classifier)
base_classifier = DecisionTreeClassifier()

# إعداد Bagging Classifier
bagging_classifier = BaggingClassifier(base_estimator=base_classifier, n_estimators=10, random_state=42)

# إعداد عملية Cross-Validation باستخدام KFold
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# تنفيذ عملية Cross-Validation وحساب النتائج
cv_scores = cross_val_score(bagging_classifier, X, y, cv=cv, n_jobs=-1)
print("===========================================")
print("Cross Vaildation ==>(Bagging classifier )")
# حساب متوسط درجات Cross-Validation
average_accuracy = sum(cv_scores) / len(cv_scores)
print("Cross-Validation Average Accuracy:", average_accuracy)

# تحويل قيم y إلى أرقام (0 و 1)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ... باقي الكود الخاص بتحميل وتنظيف البيانات

# ... باقي الكود الخاص بإعداد X و y

# ... باقي الكود الخاص بإعداد النموذج الفردي و Bagging Classifier

# تنفيذ عملية Cross-Validation والتنبؤ باستخدام cross_val_predict
y_pred = cross_val_predict(bagging_classifier, X, y_encoded, cv=cv, n_jobs=-1)

# حساب مصفوفة الارتباك
conf_matrix = confusion_matrix(y_encoded, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# توقعات الاحتمالية للفئة الموجبة
y_pred_prob = cross_val_predict(bagging_classifier, X, y_encoded, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]

# حساب منحنى التشغيل ومساحة منحنى التشغيل
fpr, tpr, thresholds = roc_curve(y_encoded, y_pred_prob)
roc_auc = auc(fpr, tpr)
print("ROC Curve AUC:", roc_auc)
print("===========================================")
# رسم منحنى التشغيل
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %2.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

