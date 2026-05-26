# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages using import statements.
2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Print all the outputs.
6. End the Program. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Priyadharshini V
RegisterNumber: 212225230219
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv("spam (1).csv", encoding="latin-1")

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("Data")
print(df.head())
print()
print(df.tail())
print()
print("Shape :", df.shape)

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = CountVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()

model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    target_names=['ham', 'spam']
)

print("\nConfusion Matrix")
print(cm)

print("\nAccuracy")
print(acc)

print("\nClassification Report")
print(report)
```

## Output:
<img width="821" height="722" alt="Screenshot 2026-05-26 170613" src="https://github.com/user-attachments/assets/08e40c11-93be-432e-9dea-c274c01347fd" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
