import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# --- Load Data ---
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0   # fake news
true["label"] = 1   # real news

data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)   # shuffle data

# --- Split ---
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- TF-IDF ---
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- Model ---
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# --- Accuracy ---
preds = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, preds))

# --- Save model and vectorizer ---
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
