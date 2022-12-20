import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn

from transformers import BertTokenizer, AutoModelForSequenceClassification, pipeline
import torch

output_dir = "../../results/models/BERT/"

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print('using ', device)

    df = pd.read_csv("../../results/data/emotions.csv")

    X = df["text"].values
    X = X.astype(str)
    y = df.iloc[:, 2:].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print('loaded validation data')
    print(len(X_test))

    model_name = 'original'  # 'ekman'

    tokenizer = BertTokenizer.from_pretrained(f"monologg/bert-base-cased-goemotions-{model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(f"monologg/bert-base-cased-goemotions-{model_name}",
                                                               num_labels=28)

    goemotions = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-classification",
        return_all_scores=True,
        function_to_apply='sigmoid',
        device=device,
        use_fast=True,
    )

    print('loaded model')


    predict = goemotions(list(X_test), batch_size=128)

    print('inferred, processing')

    y_pred = list(predict)

    y_true = list(y_test)


    def genConfusion(y_pred, y_true, classes):
        row_sums = [0 for i in range(classes)]

        matrix = [[0 for i in range(classes)] for i in range(classes)]
        for i in range(len(y_pred)):
            for x in range(classes):
                for y in range(classes):
                    p = y_pred[i][y]['score']
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
