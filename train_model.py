import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Source of dataset
# https://github.com/justmarkham/pycon-2016-tutorial/blob/master/data/sms.tsv
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_table(url, header=None, names=["label", "text"])
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(df.text, df.label_num, test_size=0.2, random_state=42)

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(solver='liblinear'))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

joblib.dump(model, 'spam_model.joblib')
print("Model saved as 'spam_model.joblib'")