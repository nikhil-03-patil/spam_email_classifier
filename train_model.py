# MAIN CODE
# import pandas as pd
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# # import preprocess

# # preprocess.clean_dataset('spam.csv', 'cleaned_email_data.csv')

# df = pd.read_csv("cleaned_email_data.csv", encoding="latin-1")
# df['clean_text'] = df['clean_text'].fillna("")
# #feature extraction (TF-DF)
# vectorizer = TfidfVectorizer(max_features=5000) #convert text into numerical features
# X = vectorizer.fit_transform(df['clean_text'])
# y = df['label']
# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #train logistic reg model
# model = LogisticRegression()
# model.fit(X_train, y_train)
# #evaluate model
# y_pred = model.predict(X_test)
# print(f"Accuracy score: {accuracy_score(y_test,y_pred) * 100:.2f}%")
# print(classification_report(y_test,y_pred))
# # Save Model & Vectorizer
# with open("models/spam_classifier.pkl", "wb") as f:
#     pickle.dump(model, f)
# with open("models/vectorizer.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)
# print("✅ Model and vectorizer saved successfully!")

# #display first rows
# print(df.head())
#check for missing values
# print("\nmissing values:",df.isnull().sum())
#check for class distribution
# print("\nclass distribution:",df['v1'].value_counts()) #v1 contains 'spam/ham' labels.

#ATTEMPT 3

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack

# Load cleaned dataset
df = pd.read_csv("cleaned_email_data.csv", encoding="latin-1")
df['clean_text'] = df['clean_text'].fillna("")

# TF-IDF on cleaned text
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(df['clean_text'])

# Combine with custom phishing-aware features
X_custom = df[['url_count', 'capital_word_count', 'spammy_word_count']].fillna(0)
X_combined = hstack([X_text, X_custom])  # Combine TF-IDF features with manual ones

y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Accuracy score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open("models/spam_classifier.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")





