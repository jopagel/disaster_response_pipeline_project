import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    """Loads the two dataframes from csv and merges to one df

    Parameters
    ----------
    messages_filepath : str
            The filepath of the messages file
    categories_filepath : str
            The filepath of the categories file

    Returns
    ----------
    df : DataFrame
            Dataframe consisting of messages and categories

    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id")
    return df


def clean_data(df):
    """Cleans the dataframe to be ready for training

    Parameters
    ----------
    df : DataFrame
            The dataframe to be cleaned

    Returns
    ----------
    df : DataFrame
            The cleaned dataframe

    """
    categories = categories["categories"].str.split(";", expand=True)
    row = categories.iloc[0, :]
    category_colnames = list(row.str.slice(start=0, stop=-2))
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.slice(start=-1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df.drop("categories", axis=1, inplace=True)
    df = df.merge(categories, left_index=True, right_index=True)
    df.message.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """Saves the clean dataframe in the database

    Parameters
    ----------
    df : DataFrame
            The filepath of the messages file
    database_filename: str
            The filename of the database

    """
    engine = create_engine(f"sqlite://{database_filename}")
    df.to_sql("data", engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
