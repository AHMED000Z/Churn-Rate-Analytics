# 🧠 Customer Churn Prediction System

## 🚀 Overview
A comprehensive end-to-end **Customer Churn Prediction System** that uses machine learning to identify customers likely to churn. The solution includes:

- Data exploration & preprocessing  
- Model development & evaluation  
- Multiple deployment interfaces for testing and production  

---

## 📚 Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
---

## 📝 Project Overview
This system helps businesses identify customers who are at risk of leaving (churn). It enables companies to proactively engage and retain these customers using machine learning insights.

The solution combines **data engineering**, **ML modeling**, and **frontend/backend web development** into a single deployable system.

---

## 🏗️ System Architecture
The system is modular and consists of the following layers:

- **Data Layer**:  
  SQL database with preprocessed views for analytics and model training  
- **Analytics Layer**:  
  Power BI dashboard for business insights and visualization  
- **Model Layer**:  
  ML pipeline for model comparison and selection  
- **API Layer**:  
  FastAPI-based backend for serving predictions  
- **Frontend Layer**:
  - Streamlit app for internal testing  
  - Web app (HTML/CSS/JavaScript) for end-users  

---

## 🌟 Features
- ✅ Data Exploration & SQL-based Preprocessing  
- 🔄 Multi-Model Training & Evaluation  
- 📊 Power BI Business Dashboards  
- 🌐 Dual Frontends: Streamlit & Responsive Web Interface  
- 🔌 RESTful API using FastAPI  
- 📈 Feature Importance Visualization  
- 📱 Mobile-Friendly Web Design  

---

## 🛠️ Technologies Used

### 📦 Data Processing & Storage:
- SQL Server / MySQL  
- Pandas  

### 🤖 Machine Learning:
- Scikit-learn  
- XGBoost  
- Random Forest (final model)  
- Joblib (for model serialization)  

### ⚙️ Backend:
- FastAPI  
- Uvicorn  

### 🖥️ Frontend:
- Streamlit  
- HTML, CSS, JavaScript  
- Responsive Web Design  

### 📈 Visualization:
- Power BI  
- Matplotlib / Seaborn  
- Dynamic Web Charts  

---

## 🔁 Project Workflow

### 🔍 Data Exploration & Preprocessing
- Explored telecom dataset using SQL  
- Created database views for analytics  
- Identified key features & correlations  
- Handled missing values  
- Categorical encoding  

### 🧠 Model Development
- Trained models:  
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
  - K-Nearest Neighbors  
  - Neural Network (MLP)  
- Hyperparameter tuning with `GridSearchCV`  
- Selected **Random Forest** as the best model  
- Saved as `best_model.pkl`  

### 📊 Power BI Dashboard
- Created visual dashboard for stakeholders  
- Visualized churn trends & customer segmentation  
- Enabled drill-down analysis  

### 💻 Streamlit Interface
- Interactive Streamlit app for testing  
- Input features dynamically  
- Show predictions & feature impact  

### 🌐 Production Web App
- FastAPI for backend logic  
- Responsive HTML/CSS/JavaScript frontend  
- Validations & error handling  
- User-friendly form inputs  

---

## ⚙️ Installation & Setup

### 🧾 Prerequisites
- Python 3.8+  
- SQL Server or MySQL  
- Power BI Desktop 