import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

TEST_SIZE = 0.2
RANDOM_STATE = 7
MAX_ITER = 50
MAX_DF = 0.7

def load_data(filepath):
    df = pd.read_csv(filepath, engine="python", on_bad_lines="skip")
    return df['text'], df.label

def train(X_train, y_train):
    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(stop_words='english', max_df=MAX_DF)
    tfidf_train = vectorizer.fit_transform(X_train)

    # Train Classifier
    model = SGDClassifier(max_iter=MAX_ITER, loss='modified_huber')
    model.fit(tfidf_train, y_train)

    return model, vectorizer

def evaluate(model, vectorizer, X_test, y_test):
    tfidf_test = vectorizer.transform(X_test)
    y_pred = model.predict(tfidf_test)

    score = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    print(f"Accuracy: {score:.4f}")
    print(f"Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    X, y = load_data('news.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model, vectorizer = train(X_train, y_train)
    evaluate(model, vectorizer, X_test, y_test)