import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report

nltk.download(["punkt", "wordnet"])


def load_data(database_filepath):
    """Loads the clean dataframe from the database

     Parameters
     ----------
     database_filepath: str
             The path of the database

     Returns
    ----------
    X : DataFrame
            The independend variable containing the messages
    y : DataFrame
            The dependend variables containing the categories
    category_names: list
            The column names of y
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("data", engine)
    X = df.iloc[:, 1]
    y = df.iloc[:, 4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """creates tokens from text messages

     Parameters
     ----------
     text: str
             The text to be tokenized

     Returns
    ----------
    clean_tokens : list
            The lemmatized tokens extracted from the text message

    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """creates tokens from text messages

     Parameters
     ----------
     text: str
             The text to be tokenized

     Returns
    ----------
    clean_tokens : list
            The lemmatized tokens extracted from the text message

    """
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )
    parameters = {
        "vect__max_features": (None, 5000),
        "clf__estimator__n_estimators": [50, 200],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """evaluates trained model with use of the test data

    Parameters
    ----------
    model: .pickle
            The trained RandomForest
    X_test: DataFrame
            The test split of independend variable containing the messages
    Y_test: DataFrame
            The test split of the dependend variables containing the categories
    category_names: list
        The column names of y
    """
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(f"Classification Metrics for column {Y_test.columns[i]}")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """save the trained model as pickle

    Parameters
    ----------
    model: .pickle
            The trained RandomForest
    model_filepath: str
            The path to save the model at
    """
    Pkl_Filename = model_filepath

    with open(Pkl_Filename, "wb") as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
