import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load datasets
print("⏳ Loading datasets...")
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# 2. Add labels
fake["label"] = 1  # Fake News
true["label"] = 0  # Real News

# 3. Combine Title and Text (Yeh sabse zaroori change hai!)
# Hum title aur text ko jod rahe hain taaki model ko zyada context mile
fake["content"] = fake["title"] + " " + fake["text"]
true["content"] = true["title"] + " " + true["text"]

# 4. Combine both datasets
df = pd.concat([fake, true], axis=0)

# 5. Text Cleaning Function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

print("🧹 Cleaning data... (Isme thoda time lagega)")
df["content"] = df["content"].apply(wordopt)

# 6. Split Features & Target
X = df["content"]  # Ab hum combined content use kar rahe hain
y = df["label"]

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 8. Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 9. Model Training
print("🤖 Training Logistic Regression Model...")
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 10. Evaluation
print("📊 Evaluating Model...")
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# 11. Save Model & Vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Success! Model & Vectorizer saved.")