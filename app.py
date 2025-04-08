# from flask import Flask, render_template, request
# import pickle
# import re
# import nltk
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import CountVectorizer

# # Ensure required NLTK resources are downloaded
# nltk.download('stopwords')

# # Load model and vectorizer with corrected file paths
# model = pickle.load(open('models/spam_classifier.pkl', 'rb'))  
# vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# app = Flask(__name__)

# def preprocess_text(text):
#     """Clean and preprocess the input text."""
#     text = re.sub(r'\W', ' ', text)  # Remove special characters
#     text = text.lower()  # Convert to lowercase
#     text = text.split()  # Tokenize
#     stop_words = set(stopwords.words('english'))  # Load stopwords
#     text = [word for word in text if word not in stop_words]  # Remove stopwords
#     return ' '.join(text)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     prediction = None
#     if request.method == 'POST':
#         email_text = request.form.get('email_text', '')  # Handle missing form input
#         if email_text.strip():  # Ensure non-empty input
#             processed_text = preprocess_text(email_text)
#             transformed_text = vectorizer.transform([processed_text])
#             result = model.predict(transformed_text)[0]
#             prediction = "Spam" if result == 1 else "Not Spam"
    
#     return render_template('index.html', prediction=prediction)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from scipy.sparse import hstack, csr_matrix

# Ensure required NLTK resources are downloaded
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open('models/spam_classifier.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

app = Flask(__name__)

# Function to clean and preprocess the input text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = text.split()
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

# Function to extract 3 additional custom features
def extract_custom_features(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    url_count = len(re.findall(r'http[s]?://\S+', text))
    capital_word_count = sum(1 for word in text.split() if word.isupper())
    spammy_words = ['prize', 'winner', 'free', 'win', 'claim', 'urgent', 'lottery']
    spammy_word_count = sum(1 for word in text.lower().split() if word in spammy_words)

    return [url_count, capital_word_count, spammy_word_count]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        email_text = request.form.get('email_text', '')
        if email_text.strip():
            processed_text = preprocess_text(email_text)

            # Vectorize using TF-IDF
            X_tfidf = vectorizer.transform([processed_text])

            # Extract custom features
            custom_features = extract_custom_features(email_text)

            # Combine both
            X_combined = hstack([X_tfidf, csr_matrix([custom_features])])

            # Predict
            result = model.predict(X_combined)[0]
            prediction = "Spam" if result == 1 else "Not Spam"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
