import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("../../results/data/emotions.csv", index_col=['id'])
    print(df)

    names = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
             "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
             "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness",
             "surprise", "neutral"]

    labels = df[names]

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size

    labels.sum(axis=0).plot.bar()
    plt.show()

    labels.sum(axis=0).div(labels.shape[0], axis=0).plot.bar()
    plt.show()


if __name__ == "__main__":
    main()