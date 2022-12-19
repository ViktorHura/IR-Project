import pandas as pd


def main():
    one = pd.read_csv('../../data/goemotions/goemotions_1.csv', index_col='id')
    two = pd.read_csv('../../data/goemotions/goemotions_2.csv', index_col='id')
    three = pd.read_csv('../../data/goemotions/goemotions_3.csv', index_col='id')

    # merge documents
    df = pd.concat([one, two, three])
    print(df)

    # filter out unclear data
    df = df[df["example_very_unclear"] == False]

    # remove unnecessary columns
    cols_to_remove = ['author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id', 'example_very_unclear']
    for c in cols_to_remove:
        df.drop(c, inplace=True, axis=1)

    # categorise each row
    def categorise(row):
        sum = 0
        for i in range(1, len(row)-1):
            sum+=row[i]
        if sum != 0 and row['neutral'] < 0.5:
            return 'keep'
        if sum == 0 and row['neutral'] > 0.5:
            return 'keep'
        return 'delete'
    df['to-remove'] = df.apply(lambda row: categorise(row), axis=1)

    # filter out unclear data
    df = df[df["to-remove"] == 'keep']

    df.drop('to-remove', inplace=True, axis=1)

    print(df)
    df.to_csv("../../results/data/emotions.csv")


if __name__ == "__main__":
    main()