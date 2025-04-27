import numpy as np 
import pandas as pd
import pickle 
import os 
import scipy
from sklearn.linear_model import LogisticRegression
import yaml 
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

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train the Logistic Regression Model"""
    try:
        params = load_params("params.yaml")
        model_c = params["model_building"]["C"]
        model_penalty = params["model_building"]["penalty"]
        model_solver = params["model_building"]["solver"]

        clf = LogisticRegression(C=model_c, penalty=model_penalty, solver=model_solver)
        clf.fit(X_train, y_train)
        logging.info("Model training completed")
        return clf 
    except Exception as e:
        logging.error("Error during model training: %s", e)
        raise 

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info("Model saved to %s", file_path)
    except Exception as e:
        logging.error("Error occurred while saving the model: %s", e)
        raise 

def main():
    try:
        X_train = scipy.sparse.load_npz('./data/processed/train_tfidf.npz')
        y_train = pd.read_csv('./data/processed/train_labels.csv')['label'].values
        logging.info("Training data loaded")

        clf = train_model(X_train, y_train)
        
        save_model(clf, "./models/model.pkl")
    except Exception as e:
        logging.error("Failed to complete the model building process: %s", e)
        raise 

if __name__ == "__main__":
    main()



