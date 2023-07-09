import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

dataset_folder = "dataset/"

sms_files = ["SMSSpamCollection"]
yt_files = ["Youtube01-Psy.csv", "Youtube02-KatyPerry.csv", "Youtube03-LMFAO.csv", "Youtube04-Eminem.csv", "Youtube05-Shakira.csv"]



class Model:
    def __init__(self):
        self.load_data()
        self.train()

    def predict(self, text):
        text = self.preprocess(text)
        text = self.vectorizer.transform([text])

        return self.model.predict(text)[0].item()
    
    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.df["processed_text"], self.df["label"], test_size=0.2, random_state=42)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.vectorize()

        self.model = MultinomialNB()
        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_test)

        print("Model trained. Evaluation results:")
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Precision:", precision_score(self.y_test, y_pred))

        
    def load_data(self):
        youtube_df = pd.concat([pd.read_csv(dataset_folder + 'youtube/' + file) for file in yt_files])
        youtube_df = youtube_df.sample(frac=1).reset_index(drop=True)
        youtube_df["label"] = youtube_df["CLASS"].apply(lambda x: 1 if x == 1 else 0)
        youtube_df = youtube_df[["CONTENT", "label"]]
        youtube_df.columns = ["text", "label"]

        sms_df = pd.read_csv(dataset_folder + 'sms/' + sms_files[0], sep="\t", header=None)
        sms_df.columns = ["label", "text"]

        sms_df["label"] = sms_df["label"].apply(lambda x: 1 if x == "spam" else 0)

        self.df = pd.concat([youtube_df, sms_df])
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.df.drop_duplicates(inplace=True)

        self.df["processed_text"] = self.df["text"].apply(self.preprocess)


    def preprocess(self, text):
        stemmer = PorterStemmer()
        stop_words = stopwords.words("english")

        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        text = nltk.word_tokenize(text)
        text = [word for word in text if word not in stop_words]
        text = [stemmer.stem(word) for word in text]

        return " ".join(text)

    def vectorize(self):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.df["processed_text"])

        self.X_train = self.vectorizer.transform(self.X_train)
        self.X_test = self.vectorizer.transform(self.X_test)