# spam_email_classifier

🟦 Overview:
This project is a Spam Email Classifier built using Machine Learning (ML) to detect whether an email is spam or not spam. It takes an email as input and predicts whether it is a legitimate email or spam using Natural Language Processing (NLP) techniques.

🟦 Model Used:
The project uses Logistic Regression, a popular classification algorithm for text-based spam detection.
Why Logistic Regression?
•	Works well for binary classification problems (Spam or Not Spam).
•	Uses a sigmoid function to classify emails based on probabilities.
•	Effective for text classification when combined with TF-IDF (Term Frequency-Inverse Document Frequency) or Count Vectorization.
•	Provides high accuracy and interpretability in spam detection tasks.

🟦 Tech Stack:
🔹 Python (Flask for web interface)
🔹 NLTK (Natural Language Toolkit for text processing)
🔹 scikit-learn (Machine learning & feature extraction)
🔹 Flask (Backend framework for web deployment)
🔹 HTML, CSS, Bootstrap (Frontend UI design)

🟦 How to Run the Project:
1️⃣ Clone the repository
git clone https://github.com/yourusername/spam-email-classifier.git
cd spam-email-classifier

2️⃣ Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows

3️⃣ Install dependencies:
pip install -r requirements.txt

4️⃣ Run the Flask app:
python app.py
Visit http://127.0.0.1:5000/ in your browser.

✈️ How It Works:
1. Users enter email text in the provided input field.
2. The email is preprocessed (cleaning, removing stopwords, etc.).
3. The TF-IDF vectorizer converts text into numerical format.
4. The Logistic Regression model classifies it as Spam or Not Spam.
5. The result is displayed in the UI.

🔜 Future Improvements:
✅ Improve accuracy by using deep learning models like LSTMs.
✅ Implement an API for integration with email services.
✅ Add real-time spam filtering for emails.




