# MAIN CODE
# import re 
# import nltk
# import pandas as pd
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# #download necessary nltk data files
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('punkt_tab')


# #initialize tools
# stop_words=set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# def preprocess_text(text):
#     """cleans and preprocesses the email text."""
#     text = text.lower()
#     text = re.sub(r'\W',' ',text)
#     text = re.sub(r'\d+', '',text )
#     words = word_tokenize(text)
#     words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
#     return " ".join(words)

# def clean_dataset(input_file, output_file):
#     """Loads, cleans, and saves the dataset."""
#     df = pd.read_csv(input_file, encoding="latin-1")
#     # Keep only relevant columns
#     df = df[['v1', 'v2']].copy()
#     df.columns = ['label', 'text']  # Rename columns

#     # Convert labels to binary (spam = 1, ham = 0)
#     df['label'] = df['label'].map({'ham': 0, 'spam': 1})

#     # Apply text preprocessing
#     df['clean_text'] = df['text'].apply(preprocess_text)

#     # Save cleaned dataset
#     df.to_csv(output_file, index=False)
#     print(f"✅ Cleaned dataset saved as: {output_file}")

# if __name__ == "__main__":
#     clean_dataset('spam.csv', 'cleaned_email_data.csv')


#attempt 3
import re
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def extract_features(text):
    """Extract custom features for phishing detection."""
    url_count = len(re.findall(r'http[s]?://', text))
    capital_word_count = len(re.findall(r'\b[A-Z]{2,}\b', text))
    spammy_words = ['verify', 'urgent', 'account', 'login', 'suspended', 'security', 'limited', 'click here']
    spam_word_count = sum(word in text.lower() for word in spammy_words)

    return pd.Series({
        'url_count': url_count,
        'capital_word_count': capital_word_count,
        'spammy_word_count': spam_word_count
    })

def preprocess_text(text):
    """Cleans and preprocesses email content."""
    text = BeautifulSoup(text, 'html.parser').get_text()  # remove HTML
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file, encoding="latin-1")
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['clean_text'] = df['text'].apply(preprocess_text)

    # Add custom phishing-aware features
    features = df['text'].apply(extract_features)
    df = pd.concat([df, features], axis=1)

    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned dataset with features saved as: {output_file}")

if __name__ == "__main__":
    clean_dataset('spam.csv', 'cleaned_email_data.csv')
