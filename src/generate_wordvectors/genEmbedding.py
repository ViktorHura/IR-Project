import pandas as pd
import string
from nltk import word_tokenize, sent_tokenize
from gensim.models import Word2Vec


def clean_lines(text, vocab):
    clean = list()
    lines = sent_tokenize(text)

    for line in lines:
        tokens = word_tokenize(line)
        # convert to lower case
        tokens = [w.lower() for w in tokens]
        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]

        # filter out non vocab
        tokens = [w for w in tokens if w in vocab]
        clean.append(tokens)

    return clean


def main():
    # load vocab
    vocab = pd.read_csv('../../results/data/vocab.csv', index_col=0)
    vocab = set(vocab.loc[:, '0'])
    # load data
    data = pd.read_csv('../../results/data/emotions.csv', index_col='id')
    print(data)

    # extract and clean training sentences
    sentences = []

    def process(row, output):
        output += clean_lines(row[0], vocab)
    data.apply(lambda row: process(row, sentences), axis=1)

    print('Total training sentences: %d' % len(sentences))

    # train word2vec model
    model = Word2Vec(sentences, vector_size=300, window=5, workers=8, min_count=1)

    # summarize vocabulary size in model
    print('Vocabulary size: %d' % len(model.wv))
    # save model
    model.wv.save_word2vec_format('../../results/data/embedding_word2vec.txt', binary=False)


if __name__ == "__main__":
    main()