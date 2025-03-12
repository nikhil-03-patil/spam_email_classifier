import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#download necessary nltk data files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')


#initialize tools
stop_words=set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """cleans and preprocesses the email text."""
    text = text.lower()
    text = re.sub(r'\W',' ',text)
    text = re.sub(r'\d+', '',text )
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def clean_dataset(input_file, output_file):
    """Loads, cleans, and saves the dataset."""
    df = pd.read_csv(input_file, encoding="latin-1")
    # Keep only relevant columns
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'text']  # Rename columns

    # Convert labels to binary (spam = 1, ham = 0)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Apply text preprocessing
    df['clean_text'] = df['text'].apply(preprocess_text)

    # Save cleaned dataset
    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned dataset saved as: {output_file}")

if __name__ == "__main__":
    clean_dataset('spam.csv', 'cleaned_email_data.csv')

