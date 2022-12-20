import time
import pandas as pd
from itertools import chain
from transformers import BertTokenizer, AutoModelForSequenceClassification, pipeline

def main():
    model_name = 'original'  # 'ekman'

    tokenizer = BertTokenizer.from_pretrained(f"monologg/bert-base-cased-goemotions-{model_name}", model_max_len=512)
    model = AutoModelForSequenceClassification.from_pretrained(f"monologg/bert-base-cased-goemotions-{model_name}",
                                                               num_labels=28)

    goemotions = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-classification",
        return_all_scores=True,
        function_to_apply='sigmoid',
        device=0,
        binary_output=True
    )

    print("loaded model and tokenizer")

    data = pd.read_csv('../../results/data/cleanData.csv').astype(str)
    print('loaded data')

    names = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
             "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
             "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
             "remorse", "sadness", "surprise", "neutral"]
    print('predicting - timing')
    start = time.time()

    predict = goemotions(list(data['text'].values), batch_size=64, padding=True, truncation=True)

    end = time.time()
    print("timed - ", end - start)

    print('reshaping')
    start = time.time()
    a = chain.from_iterable(predict)
    b = list(a)

    predict = pd.DataFrame(b)

    predict = predict['score']

    predict = pd.DataFrame(predict.values.reshape(-1, 28), columns=names)

    predict.insert(0, "links", data['links'])
    predict.insert(0, "text", data['text'])

    end = time.time()
    print("timed - ", end - start)

    predict.to_csv("../../results/data/BERT-labelled.csv", index=False, float_format='{:f}'.format, encoding='utf-8')
    print(predict)


if __name__ == "__main__":
    main()
