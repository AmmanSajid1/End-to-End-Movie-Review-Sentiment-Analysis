import os
import pickle
import logging
import yaml
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.logger import logging

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logging.info("Data loaded and NaNs filled from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error occurred while loading the data: %s", e)
        raise

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int = 5000) -> tuple:
    """Apply TF-IDF Vectorizer to the data"""
    try:
        logging.info("Applying TF-IDF with max_features=%s", max_features)
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['review'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['review'].values
        y_test = test_data['sentiment'].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Save the fitted vectorizer for later use
        os.makedirs('models', exist_ok=True)
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        logging.info("TF-IDF vectorizer saved at 'models/vectorizer.pkl'")

        logging.info('TF-IDF applied and data transformed')
        return X_train_tfidf, y_train, X_test_tfidf, y_test
    except Exception as e:
        logging.error("Error during the TF-IDF transformation: %s", e)
        raise

def save_sparse_matrix(matrix, file_path: str):
    """Save a sparse matrix in compressed format"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        scipy.sparse.save_npz(file_path, matrix)
        logging.info("Sparse matrix saved to %s", file_path)
    except Exception as e:
        logging.error("Error saving sparse matrix: %s", e)
        raise

def save_labels(labels, file_path: str):
    """Save labels to a CSV file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        pd.DataFrame(labels, columns=["label"]).to_csv(file_path, index=False)
        logging.info("Labels saved to %s", file_path)
    except Exception as e:
        logging.error("Error saving labels: %s", e)
        raise

def main():
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        train_data = load_data("./data/interim/train_processed.csv")
        test_data = load_data("./data/interim/test_processed.csv")

        X_train_tfidf, y_train, X_test_tfidf, y_test = apply_tfidf(train_data, test_data, max_features=max_features)

        save_sparse_matrix(X_train_tfidf, "./data/processed/train_tfidf.npz")
        save_sparse_matrix(X_test_tfidf, "./data/processed/test_tfidf.npz")

        save_labels(y_train, "./data/processed/train_labels.csv")
        save_labels(y_test, "./data/processed/test_labels.csv")

        logging.info("Feature engineering completed successfully")
    except Exception as e:
        logging.error("Failed to complete the feature engineering process: %s", e)
        raise

if __name__ == "__main__":
    main()



        

