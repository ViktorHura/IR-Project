import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

INIT_LR = 0.001
EPOCHS = 3
BS = 64

def main():
    df = pd.read_csv("../../results/data/emotions.csv")

    X = df["text"].values
    X = X.astype(str)
    y = df.iloc[:, 2:].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print('loaded datasets')

    ## Tokenize words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    vocab = pd.read_csv('../../results/data/vocab.csv', index_col=0)
    vocab = set(vocab.loc[:, '0'])
    num_words = len(vocab)
    tokenizer.word_index = {e: i for e, i in tokenizer.word_index.items() if
                            i <= num_words}  # <= because tokenizer is 1 indexed
    tokenizer.word_index[tokenizer.oov_token] = num_words + 1

    ## Save tokenizer
    with open('../../results/models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    maxlen = max(
        [len(s.split()) for
         s in df["text"].values]
    )
    print('Max length: ', maxlen)

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    print(len(X_train))

    ## create embeddings
    # https://github.com/sharonchoong/covid19tweets/blob/main/Emotion%20Classification/Emotion%20Classification.ipynb
    def create_embedding_matrix(filepath, word_index, embedding_dim):
        vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
        embedding_matrix = np.zeros((vocab_size, embedding_dim))

        with open(filepath, encoding='utf-8') as f:
            for line in f:
                word, *vector = line.split()
                if word in word_index:
                    idx = word_index[word]
                    embedding_matrix[idx] = np.array(
                        vector, dtype=np.float32)[:embedding_dim]
        return embedding_matrix

    embedding_dim = 300
    embedding_matrix = create_embedding_matrix('../../data/glove/glove.6B.300d.txt', tokenizer.word_index, embedding_dim)

    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    embedding_accuracy = nonzero_elements / vocab_size
    print('embedding accuracy: ' + str(embedding_accuracy))

    def get_f1(y_true, y_pred):  # taken from old keras source code
        true_positives = tensorflow.keras.backend.sum(tensorflow.keras.backend.round(tensorflow.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tensorflow.keras.backend.sum(tensorflow.keras.backend.round(tensorflow.keras.backend.clip(y_true, 0, 1)))
        predicted_positives = tensorflow.keras.backend.sum(tensorflow.keras.backend.round(tensorflow.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tensorflow.keras.backend.epsilon())
        recall = true_positives / (possible_positives + tensorflow.keras.backend.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + tensorflow.keras.backend.epsilon())
        return f1_val

    CLASSES = 28

    ## CNN
    model = Sequential()
    model.add(
        layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False))
    model.add(layers.Conv1D(256, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(28, activation='sigmoid'))
    model.compile(optimizer=optimizers.Adam(lr=0.0002), loss='binary_crossentropy',
                  metrics=["accuracy", metrics.Precision(name="precision"), metrics.Recall(name="recall"), get_f1])
    model.summary()



    callbacks = [
                 #EarlyStopping(monitor='val_loss', patience=2),
                 ModelCheckpoint(filepath='../../results/models/checkpoints/model-checkpoint-{epoch}.h5', monitor='val_get_f1', save_best_only=False)
    ]
    fit = model.fit(X_train, y_train, epochs=EPOCHS, verbose=True, callbacks=callbacks,
                    validation_data=(X_test, y_test),
                    batch_size=BS)

    model.save('../../results/models/model.h5')

    def plot_loss_evaluation(r):
        plt.figure(figsize=(12, 8))

        plt.title('Training and Loss function')
        plt.subplot(2, 3, 1)
        plt.plot(r.history['loss'], label='loss')
        plt.plot(r.history['val_loss'], label='val_loss')
        plt.legend()
        plt.title('Training and Loss function')

        plt.subplot(2, 3, 2)
        plt.plot(r.history['accuracy'], label='accuracy')
        plt.plot(r.history['val_accuracy'], label='val_acc')
        plt.legend()
        plt.title('Accuracy')

        plt.subplot(2, 3, 3)
        plt.plot(r.history['recall'], label='recall')
        plt.plot(r.history['val_recall'], label='val_recall')
        plt.legend()
        plt.title('Recall')

        plt.subplot(2, 3, 4)
        plt.plot(r.history['precision'], label='precision')
        plt.plot(r.history['val_precision'], label='val_precision')
        plt.legend()
        plt.title('Precision')

        plt.subplot(2, 3, 5)
        plt.plot(r.history['get_f1'], label='f1')
        plt.plot(r.history['val_get_f1'], label='val_f1')
        plt.legend()
        plt.title('F1')

        plt.savefig('../../results/models/results.png')
        plt.show()

    plot_loss_evaluation(fit)
    textstats = "Epochs: " + str(EPOCHS) + "\n"
    textstats += "Batch Size: " + str(BS) + "\n"

    textstats += "Classes: " + str(CLASSES) + "\n"
    textstats += "Max length: " + str(maxlen) + "\n"

    textstats += "Loss " + str(fit.history['loss'][-1]) + "\n"
    textstats += "Validation Loss " + str(fit.history['val_loss'][-1]) + "\n"

    textstats += "Accuracy " + str(fit.history['accuracy'][-1]) + "\n"
    textstats += "Validation Accuracy " + str(fit.history['val_accuracy'][-1]) + "\n"

    textstats += "Recall " + str(fit.history['recall'][-1]) + "\n"
    textstats += "Validation Recall " + str(fit.history['val_recall'][-1]) + "\n"

    textstats += "Precision " + str(fit.history['precision'][-1]) + "\n"
    textstats += "Validation Precision " + str(fit.history['val_precision'][-1]) + "\n"

    textstats += "F1 " + str(fit.history['get_f1'][-1]) + "\n"
    textstats += "Validation F1 " + str(fit.history['val_get_f1'][-1]) + "\n"

    text_file = open("../../results/models/stats.txt", "w")
    text_file.write(textstats)
    text_file.close()


if __name__ == "__main__":
    main()


