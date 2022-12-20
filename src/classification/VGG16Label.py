import time
import tensorflow
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

def main():
    ## Tokenize words
    maxlen = 33

    with open('../../results/models/VGG16-long/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    def get_f1(y_true, y_pred):  # taken from old keras source code
        true_positives = tensorflow.keras.backend.sum(
            tensorflow.keras.backend.round(tensorflow.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tensorflow.keras.backend.sum(
            tensorflow.keras.backend.round(tensorflow.keras.backend.clip(y_true, 0, 1)))
        predicted_positives = tensorflow.keras.backend.sum(
            tensorflow.keras.backend.round(tensorflow.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tensorflow.keras.backend.epsilon())
        recall = true_positives / (possible_positives + tensorflow.keras.backend.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + tensorflow.keras.backend.epsilon())
        return f1_val

    ## CNN
    with tensorflow.keras.utils.custom_object_scope({'get_f1': get_f1}):
        model = tensorflow.keras.models.load_model('../../results/models/VGG16-long/model.h5')
        print("loaded model and tokenizer")

        data = pd.read_csv('../../results/data/cleanData.csv').astype(str)
        print('loaded data - timing')
        start = time.time()
        inpt = tokenizer.texts_to_sequences(data['text'])
        print("tokenized")
        inpt = pad_sequences(inpt, padding='post', maxlen=maxlen)
        print("padded")

        end = time.time()
        print("timed - ", end-start)

        names = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
                 "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
                 "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
                 "remorse", "sadness", "surprise", "neutral"]
        print('predicting - timing')
        start = time.time()

        predict = pd.DataFrame(model.predict(inpt, batch_size=10000), columns=names)

        predict.insert(0, "links", data['links'])
        predict.insert(0, "text", data['text'])

        end = time.time()
        print("timed - ", end - start)

        predict.to_csv("../../results/data/VGG16-labelled.csv", index=False, float_format='{:f}'.format, encoding='utf-8')
        print(predict)


if __name__ == "__main__":
    main()
