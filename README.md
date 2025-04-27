# **Sentiment Analysis: End-to-End Machine Learning Deployment with AWS EKS**

This project showcases a complete **machine learning deployment pipeline**, from **data ingestion** and **model training** to **CI/CD**, **containerization**, **Kubernetes deployment**, and **production monitoring**. Built with a real-world use case in mind, it incorporates industry-leading tools and best practices for MLOps and DevOps.

### 🚀 **Project Highlights:**

- **Full ML pipeline** from data ingestion to model deployment and monitoring.
- **Cloud-native architecture** utilizing AWS services like **EKS**, **S3**, and **EC2**.
- **MLOps tools** like **MLflow**, **DVC**, and **Dagshub** for experiment tracking and versioning.
- **CI/CD automation** using **GitHub Actions** for continuous retraining, testing, and deployment.
- **Flask API** to serve model predictions via a REST interface.
- **Production-grade deployment** on Kubernetes, ensuring scalability and availability.
- **Real-time monitoring** using **Prometheus** and **Grafana** for metrics and observability.

---

## 🧰 **Tech Stack**

| **Area**             | **Tools & Services**                   |
|----------------------|----------------------------------------|
| **Project Structure** | Cookiecutter Data Science Template     |
| **Experiment Tracking** | MLflow, DVC, Dagshub               |
| **Data/Model Storage** | AWS S3                               |
| **Model Serving**     | Flask Web API                         |
| **Containerization**  | Docker                                |
| **CI/CD**             | GitHub Actions, AWS ECR               |
| **Orchestration**     | Kubernetes (AWS EKS)                   |
| **Monitoring**        | Prometheus, Grafana (on EC2)          |
| **Cloud Services**    | AWS (IAM, EKS, EC2, S3, ECR)          |

---

## 📂 **Project Structure**

```
├── .dvc/                  # DVC cache & experiment tracking
├── .github/               # GitHub Actions CI/CD pipelines
├── data/                  # Data folder (Cookiecutter structure)
├── docs/                  # Project documentation
├── flask_app/             # Flask API for model predictions
├── local_s3/              # Local mock S3 for testing
├── logs/                  # Log files
├── models/                # Trained model artifacts
├── notebooks/             # Jupyter notebooks for exploration
├── reports/               # Generated reports & figures
├── scripts/               # Utility scripts
├── src/                   # Source code (data, features, model, etc.)
├── tests/                 # Unit tests
```

---

## 🌟 **Key Features**

- **Version Control for Data & Models:**
  - **DVC** manages datasets and model artifacts, leveraging **AWS S3** as remote storage.
  - **MLflow** tracks experiment results and model metrics, integrated with **Dagshub** for version control.

- **Automated CI/CD Pipeline:**
  - **GitHub Actions** handles model retraining, Docker image builds, and deployments to **AWS ECR** and **EKS**.

- **Production-Grade Deployment:**
  - Model served via a **Flask** app inside a **Docker container**, ensuring portability and scalability.
  - **Kubernetes** handles the deployment, scaling, availability, and load balancing.

- **Monitoring & Observability:**
  - **Prometheus** collects real-time metrics, while **Grafana** provides rich visualizations of app health and latency.

- **Infrastructure as Code:**
  - Automated **EKS** setup with `eksctl`.
  - Kubernetes manifests for app deployment and service exposure.

---

## 🛠 **Deployment Architecture**

```plaintext
Local Development ➔ GitHub Push ➔ CI/CD Pipeline ➔ Build Docker Image ➔ Push to ECR ➔ Deploy to EKS ➔ Serve Predictions
                                            ↘ Monitor Metrics ↙
                                        Prometheus & Grafana
```

---

## ⚡ **Deployment Notes**

- **AWS EKS** cluster managed via `eksctl`.
- **Prometheus** and **Grafana** deployed on **EC2** instances for monitoring.
- Customizable **monitoring endpoints** and **dashboards**.
- Cleanup scripts for **AWS resource teardown** after usage.
- Assumes familiarity with **Docker**, **Kubernetes**, **AWS CLI**, and **MLOps** concepts.

---

## 🎯 **Why This Project?**

This project not only demonstrates **machine learning deployment**, but also highlights the end-to-end **system architecture** required for building, serving, monitoring, and continuously updating production ML models. The focus is on **real-world scalability**, **robust automation**, and **observability**—critical skills in modern ML and DevOps environments.

---