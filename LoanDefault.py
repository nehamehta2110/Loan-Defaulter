import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def read_from_csv(filename):
    """
    reads from input csv
    """
    df = pd.read_csv(filename)
    print(df.head)
    return df


def default_label(value):
    """
    Label 0 if not defaulter, else Label 1
    """
    if value > 30:
        return 1
    else:
        return 0


def labeling(df):
    """
    labeling the dataframe
    """
    df['class'] = df['max_dpd'].apply(default_label)
    df.sort_values(by=['master_user_id'], inplace=True)
    print(df.head(20))
    return df


def analyse(df):
    """
    feature engineering
    """
    df.describe()
    correlation_matrix = df.corr()
    print(correlation_matrix)
    return correlation_matrix


def visualize(cm):
    """
    visualizing features
    """
    sns.heatmap(cm, annot=True)
    plt.show()


def classifier(df):
    """
    class
    """
    X = df.drop(columns=['class', 'disbursed_at', 'max_dpd'])
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))


def main():
    """
    main execution
    """
    frame = read_from_csv('./data_loans_5k.csv')
    frame1 = labeling(frame)
    cor = analyse(frame1)
    visualize(cor)
    classifier(frame1)


if __name__ == '__main__':
    main()


