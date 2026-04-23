import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

df = pd.read_csv('spam_ham_dataset.csv')
print(df.head())

X = df['text']
y = df['label']

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#[5 rows x 4 columns]
#Accuracy: 0.8582474226804123
#             precision    recall  f1-score   support
#
#         ham       0.83      1.00      0.91      1109
#        spam       1.00      0.50      0.67       443
#
#    accuracy                           0.86      1552
#   macro avg       0.92      0.75      0.79      1552
#weighted avg       0.88      0.86      0.84      1552

#[[1109    0]
# [ 220  223]]