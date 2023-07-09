# Spam/Ham classifier

This is an AI as a API project. It can be used to classify a text as spam or ham. Since, it is in REST API form, it can be used in any programming language.

## Plan of action

In this section, I will explain how I approached this problem.

### Data

I used two datasets for this and they are:

- [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
- [YouTube Spam Collection Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/images)

Both the datasets where combined and used for training and testing.

### EDA

I did some basic EDA on the data and found out that the SMS data is imbalanced. And although the YouTube data is balanced, it was from music videos, meaning that it might not be an ideal representation of real world data. Apart from this I was not able to find any other insights from the data. Lastly, I decided to use both the datasets.

### Preprocessing

In preprocessing, I did the following:

- Converted all the text to lowercase
- Removed all the stopwords
- Kept only alphanumeric characters
- Lemmatized the words

### Vectorization

To vectorize the data, I tried both TF-IDF and Count Vectorizer. I found out that TF-IDF performed better than Count Vectorizer. So, I used TF-IDF for vectorization.

### Model

I mostly tried 'naive_bayes' and it's variation like 'multinomial_nb', 'bernoulli_nb', and 'gaussian_nb'. I found out that 'multinomial_nb' performed better than others. So, I used 'multinomial_nb' for the model. In future, I might try other models like 'svm', 'logistic_regression', etc.

### API

I decided to go with FastAPI for the API. It is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints. It is one of the fastest Python frameworks available. Note: although you might see the final code in this repository, I have tested different approaches on a separate juptyer notebook. Then used the best approach for the final code.

## How to use

### Installation

First, clone the repository.

```bash
git clone
```

Then, install the dependencies.

```bash
pip install -r requirements.txt
```

Note: You might need to install nltk stopwords and punkt manually for the first time.

```bash
python -m nltk.downloader stopwords
python -m nltk.downloader punkt
```

### Running the API

To run the API, run the following command:

```bash
python main.py
```

Once the API is running, you can go to http://localhost:8000/predict/you%20won%201000$ to see the result. Ideally, you should see the following result:

```json
{
  "text": "you won 1000$",
  "label": "spam"
}
```

## Future work

- Try other models like 'svm', 'logistic_regression', etc.
- Try other vectorization techniques like 'word2vec', 'glove', etc.
- Try other datasets like [Twitter Spam Collection Dataset](https://www.kaggle.com/cosmos98/twitter-and-reddit-sentimental-analysis-dataset).
- Make sure it fits real world data.

## References

- [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
- [YouTube Spam Collection Dataset](https://www.kaggle.com/datasnaek/youtube)
- [FastAPI](https://fastapi.tiangolo.com/)
- [nltk](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)

## Conclusion

Although, This is beginner project in the field of NLP, I learned a lot while doing this project. I hope you learned something too. If you have any questions, suggestions, or feedback, feel free to contact me. Thank you for reading this. Have a nice day!

## Contact

- [Email](mailto:dakshesh.jain14@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/dakshesh-jain)
- [Twitter](https://twitter.com/_dakshesh)
- [GitHub](https://github.com/dakshesh14)
- [Kaggle](https://www.kaggle.com/daksheshjain)
- [Portfolio](https://dakshesh.me)
