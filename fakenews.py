from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Sample training data (very simple for demo)
texts = [
    "The government passed a new healthcare reform bill today.",
    "Aliens have landed and taken over the White House.",
    "NASA launches new mission to study Mars surface.",
    "Celebrity cloned by secret government lab."
]
labels = [0, 1, 0, 1]  # 0 = Real, 1 = Fake

# Vectorize and train
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

# Save the model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Saved: fake_news_model.pkl & tfidf_vectorizer.pkl")
