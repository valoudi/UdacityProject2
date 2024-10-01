import sys
import pandas as pd 
import sqlalchemy 

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge message and category data from CSV files.

    Args:
        messages_filepath (str): Filepath to the CSV file containing messages.
        categories_filepath (str): Filepath to the CSV file containing categories.

    Returns:
        pandas.DataFrame: Merged dataset containing messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Clean the merged dataset by removing duplicates, extracting category names,
    converting category values to binary, and dropping the original categories column.

    Args:
        df (pandas.DataFrame): The merged dataset containing messages and categories.

    Returns:
        pandas.DataFrame: The cleaned dataset with binary category columns.
    """
    
    
    
    # First, we extract category names from the categories column
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]  # We get the first row to extract category names
    category_colnames = row.apply(lambda x: x.split('-')[0])  # We split by '-' to keep category names
    categories.columns = category_colnames  # We rename columns
    
    # Now we convert the type of columns to boolean
    for column in categories:
        # Set each value to be either 0 or 1
        categories[column] = categories[column].apply(lambda x: 1 if int(x.split('-')[1]) >= 1 else 0).astype(bool)
    
    df.drop('categories', axis=1, inplace=True)  # We drop the original 'categories' column from df

    df = pd.concat([df, categories], axis=1)  # We concatenate df and categories
    

    # To finish, we remove duplicate rows and constant columns
    df.drop_duplicates(inplace=True)
    df = df.loc[:,df.apply(pd.Series.nunique) != 1]
    return df 


def save_data(df, database_filename):
    """
    Save the cleaned data to a SQLite database.

    Args:
        df (pandas.DataFrame): The cleaned data to be saved.
        database_filename (str): The filepath to save the SQLite database.

    Returns:
        None
    """
    engine = sqlalchemy.create_engine('sqlite:///' + database_filename)
    df.to_sql('T_DISASTER_RESPONSE', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()