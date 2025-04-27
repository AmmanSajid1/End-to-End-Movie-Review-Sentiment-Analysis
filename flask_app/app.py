from flask import Flask, render_template, request
from logger import logging
import mlflow
import pickle 
import os 
import pandas as pd 
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub

import warnings 
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def lemmatization(text):
    "Lemmatize the text"
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text"""
    text = [word for word in text.split() if word not in stop_words]
    return " ".join(text)

def remove_numbers(text):
    "Remove numbers from the text"
    text = ''.join([char for char in text if not char.isdigit()])
    return text 

def lower_case(text):
    "Convert text to lower case"
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def remove_punctuation(text):
    """Remove punctuation from the text"""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace(":", "")
    text = re.sub("\s+", " ", text).strip()
    return text 

def remove_urls(text):
    """Remove URLS from text"""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub(r'', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_urls(text)
    text = lemmatization(text)

    return text 

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/AmmanSajid1/End-to-End-Movie-Review-Sentiment-Analysis.mlflow')
# dagshub.init(repo_owner='AmmanSajid1', repo_name='End-to-End-Movie-Review-Sentiment-Analysis', mlflow=True)
# -------------------------------------------------------------------------------------

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "AmmanSajid1"
repo_name = "End-to-End-Movie-Review-Sentiment-Analysis"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Initialize Flask app
app = Flask(__name__)

# Create a custom registry
registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)

PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# Model and vectorizer setup
model_name = "my_model"
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Staging"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None 

model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
logging.info(f"Fetching latest model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)
logging.info("Model loaded successfully")
vectorizer = pickle.load(open("./models/vectorizer.pkl", "rb"))

# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response 

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form["text"]
    if not text.strip():
        logger.warning("Empty input received from user.")
        return render_template("index.html", result="Please enter some text for prediction.")

    # clean text
    text = normalize_text(text)
    # Convert to features 
    features = vectorizer.transform([text])
    # Predict
    logging.info("Making prediction")
    result = model.predict(features)
    prediction = result[0]

    # Increment prediction metric
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

    # Measure latency
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction)

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    # app.run(debug=True) # for local use
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker


