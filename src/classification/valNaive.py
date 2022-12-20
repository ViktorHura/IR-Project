import pandas as pd
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
import pickle

tokenizer_path = "../../results/models/naive-short/tokenizer.pickle"
model_path = "../../results/models/naive-short/model.h5"
output_dir = "../../results/models/naive-short/"

def main():
    df = pd.read_csv("../../results/data/emotions.csv")

    X = df["text"].values
    X = X.astype(str)
    y = df.iloc[:, 2:].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ## Tokenize words
    maxlen = 33

    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

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
        model = tensorflow.keras.models.load_model(model_path)
        predict = model.predict(X_test)

        y_pred = list(predict)

        y_true = list(y_test)

        def genConfusion(y_pred, y_true, classes):
            row_sums = [0 for i in range(classes)]

            matrix = [[0 for i in range(classes)] for i in range(classes)]
            for i in range(len(y_pred)):
                for x in range(classes):
                    for y in range(classes):
                        p = y_pred[i][y]
                        l = y_true[i][x]
                        if p >= 0.5:
                            matrix[x][y] += p * l
                            row_sums[x] += p * l

            for x in range(classes):
                for y in range(classes):
                    matrix[x][y] /= row_sums[x]

            names = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
                     "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
                     "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
                     "remorse", "sadness", "surprise", "neutral"]

            df_cm = pd.DataFrame(matrix, names[0:classes], names[0:classes])
            print(df_cm)
            plt.figure(figsize=(20, 20))
            sn.set(font_scale=1.4)  # for label size
            s = sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})  # font size
            s.set(xlabel='Predicted', ylabel='Label')

            if classes == 27:
                plt.savefig(output_dir + 'confuse.png')
            else:
                plt.savefig(output_dir + 'confuse-neutral.png')
            plt.show()

        genConfusion(y_pred, y_true, 27)
        genConfusion(y_pred, y_true, 28)


if __name__ == "__main__":
    main()
