# from flask import Flask, request, render_template,jsonify
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer

# app = Flask(__name__)

# #load the trained model and vectorixer
# with open("models/spam_classifier.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("models/vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/", methods=["POST"])
# def predict():
#     data = request.form["email_text"]

#     #preprocess and vectorize the input
#     vectorized_data = vectorizer.transform([data])

#     # predict spam or not
#     prediction = model.predict(vectorized_data)[0]
#     result = "Spam" if prediction==1 else "Not Spam"

#     return jsonify({"prediction": result})

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Ensure required NLTK resources are downloaded
nltk.download('stopwords')

# Load model and vectorizer with corrected file paths
model = pickle.load(open('models/spam_classifier.pkl', 'rb'))  
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

app = Flask(__name__)

def preprocess_text(text):
    """Clean and preprocess the input text."""
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize
    stop_words = set(stopwords.words('english'))  # Load stopwords
    text = [word for word in text if word not in stop_words]  # Remove stopwords
    return ' '.join(text)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        email_text = request.form.get('email_text', '')  # Handle missing form input
        if email_text.strip():  # Ensure non-empty input
            processed_text = preprocess_text(email_text)
            transformed_text = vectorizer.transform([processed_text])
            result = model.predict(transformed_text)[0]
            prediction = "Spam" if result == 1 else "Not Spam"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
