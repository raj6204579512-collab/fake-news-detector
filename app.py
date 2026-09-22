from flask import Flask, render_template, request
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download("stopwords")

# --------------------------------
# Flask App
# --------------------------------
app = Flask(__name__)

# --------------------------------
# Load Saved Model and TF-IDF
# --------------------------------
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# --------------------------------
# Text Preprocessing
# --------------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Tokenization
    words = text.split()

    # Stopwords + stemming
    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# --------------------------------
# Home Page
# --------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------
# Prediction
# --------------------------------
@app.route("/", methods=["POST"])
def predict():

    news = request.form["news"]

    # Preprocess news
    cleaned_news = preprocess_text(news)

    # TF-IDF
    vector = tfidf.transform([cleaned_news]).toarray()

    # Prediction
    prediction = model.predict(vector)

    # Confidence
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        confidence = round(max(probabilities) * 100, 2)

    # Label Mapping
    if prediction[0] == 0:
        result = "✅ Real News"
    else:
        result = "❌ Fake News"

    # Return result
    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        news=news
    )


# --------------------------------
# Run Application
# --------------------------------
if __name__ == "__main__":
    app.run(debug=True)
