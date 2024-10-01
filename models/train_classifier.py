import sys
import pandas as pd 
import sqlalchemy 
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')  
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
from sklearn.model_selection import  train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report



def load_data(database_filepath):
    """
    Load data from a SQLite database.
    Args:
        database_filepath (str): Filepath to the SQLite database.

    Returns:
        tuple: A tuple containing the following elements:
            - X (pandas.DataFrame): Features dataset.
            - Y (pandas.DataFrame): Target dataset.
            - category_names (list): List of category names.
    """
    # Load data from database
    engine = sqlalchemy.create_engine(f'sqlite:///{database_filepath}')

    with engine.begin() as conn:
        query = sqlalchemy.text("""SELECT * FROM T_DISASTER_RESPONSE""")
        df = pd.read_sql_query(query, conn)

    # Split features and target
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    # Get category names
    category_names = Y.columns.tolist()

    return X, Y, category_names

def tokenize(text):
    """
    Tokenize and lemmatize text data.
    Args:
        text (str): Text data to be tokenized and lemmatized.
        
    Returns:
        list: A list of cleaned and lemmatized tokens.
    """
    # Convert text to lowercase and remove leading/trailing whitespace
    text = text.lower().strip()
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens




def build_model():
    """
    Build a machine learning pipeline for text processing and classification with GridSearch for hyperparameter tuning.

    Returns:
        Pipeline: A scikit-learn Pipeline object consisting of a CountVectorizer, TfidfTransformer, and a MultiOutputClassifier with GridSearchCV.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier())) 
    ], verbose=True)

    # Define the parameter grid for GridSearchCV
    parameters = {
        'clf__estimator__n_estimators': [5, 10, 15],
        'clf__estimator__learning_rate': [0.5, 1, 1.5],
    }

    # Create the GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, cv = 2 ) #Must be cv = 5 but my computer is really slow :) 
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the trained model and generate a classification report.
    Args:
        model (object): The trained model object.
        X_test (pandas.DataFrame): Test features dataset.
        Y_test (pandas.DataFrame): Test target dataset.
        category_names (list): List of category names.
    """
    y_pred = model.predict(X_test)

    # Print the classification report for each category
    print('Classification Report:')
    print(classification_report(Y_test.values, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.
    Args:
        model (object): The trained model object.
        model_filepath (str): Filepath to save the model.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()