import pandas as pd
from collections import Counter
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def clean_entry(text):
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [token for token in stripped if token.isalpha()]
    # filter out stop words
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def main():
    data = pd.read_csv('../../results/data/emotions.csv', index_col='id')
    print(data)

    vocab = Counter()

    def process(row):
        tokens = clean_entry(row['text'])
        vocab.update(tokens)

    data.apply(lambda row: process(row), axis=1)

    print(len(vocab))

    min_occurane = 2
    tokens = pd.Series([k for k, c in vocab.items() if c >= min_occurane])
    print(tokens)
    tokens.to_csv('../../results/data/vocab.csv')


if __name__ == "__main__":
    main()