from flask import Flask, render_template, request
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --------------------------------
# Download stopwords
# --------------------------------
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

    # Convert to lowercase
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

    # Remove stopwords and stemming
    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# --------------------------------
# Home Page
# --------------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "GET":
        return render_template("index.html")

    # --------------------------------
    # Get News
    # --------------------------------
    news = request.form["news"]

    # --------------------------------
    # Preprocess News
    # --------------------------------
    cleaned_news = preprocess_text(news)

    # --------------------------------
    # Convert Text into TF-IDF Vector
    # --------------------------------
    vector = tfidf.transform([cleaned_news]).toarray()

    # --------------------------------
    # Make Prediction
    # --------------------------------
    prediction = model.predict(vector)

    # --------------------------------
    # Confidence
    # --------------------------------
    confidence = None
    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        confidence = round(max(probabilities) * 100, 2)

    # --------------------------------
    # Debug Information
    # --------------------------------
    print("=" * 60)

    print("Original News:")
    print(news)

    print()

    print("Cleaned News:")
    print(cleaned_news)

    print()

    print("Model Classes:")
    if hasattr(model, "classes_"):
        print(model.classes_)

    print()

    print("Prediction:")
    print(prediction[0])

    if probabilities is not None:
        print()

        print("Prediction Probabilities:")
        print(probabilities)

    print("=" * 60)

    # --------------------------------
    # Label Mapping
    #
    # 0 = Real News
    # 1 = Fake News
    # --------------------------------
    if prediction[0] == 0:
        result = "✅ Real News"
    else:
        result = "❌ Fake News"

    # --------------------------------
    # Send Result to HTML
    # --------------------------------
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
