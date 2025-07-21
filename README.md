# ☁️ Climate-Responsive Cloud Resource Management

This project presents an intelligent and sustainable cloud infrastructure that integrates real-time climate data with dynamic cloud resource scaling. It uses machine learning and AWS automation to optimize cooling and compute resource allocation in data centers—especially in temperature-sensitive regions like Bengaluru, India.

---

## 🌍 Project Title

**Leveraging Climate Data for Sustainable Cloud Resource Allocation**

---

## 🎯 Objectives

- Forecast environmental conditions (temperature, humidity) using NASA POWER dataset.
- Predict upcoming temperature trends with ML models (XGBoost, Random Forest).
- Implement **adaptive cloud scaling** based on:
  - 🔥 Climate data (e.g., high temperature → scale-up)
  - ⚙️ CPU utilization
- Use **AWS CloudWatch, Lambda, S3, and EC2** for autoscaling infrastructure.
- Apply **linear programming** for optimized power and cooling resource allocation.

---

## 🛠️ Tech Stack

| Layer            | Technologies Used                                |
|------------------|--------------------------------------------------|
| Data Source       | NASA POWER Climate Data API                     |
| Machine Learning  | XGBoost, Random Forest, Linear Regression        |
| Cloud Infrastructure | AWS EC2, S3, SageMaker, Lambda, CloudWatch |
| Automation        | AWS Auto Scaling Groups, SNS notifications      |
| Optimization      | Linear Programming (cooling/power distribution) |

---

## 🧠 Methodology

### 1. 🔍 Climate Data Collection
- Source: NASA POWER API
- Focused on historical and forecasted climate data for Bengaluru
- Collected features: Temperature, Humidity, Wind Speed, Solar Radiation

### 2. 🧹 Data Preprocessing
- Missing value imputation
- Feature selection based on cooling influence
- Data normalization and splitting into training/testing

### 3. 📈 Temperature Forecasting
- Trained models: **XGBoost**, **Random Forest**, **Linear Regression**
- Evaluated on MAE, MSE, R² Score
- **Best performing model**: XGBoost (MAE: 0.052, R²: 0.99)

### 4. ⚙️ CPU Utilization-Based Scaling
- AWS **CloudWatch Alarms** monitor EC2 CPU usage
- Triggers auto-scale:
  - ↑ Add EC2 instances if CPU > 80%
  - ↓ Remove instances if CPU < 30%
- Notifications sent via **AWS SNS**

### 5. 🌡️ Temperature-Based Scaling
- Climate forecasts stored in **AWS S3**
- Lambda reads forecasts and pushes custom metrics to CloudWatch
- Auto-scaling rules:
  - > 35°C → Scale Up (to handle cooling loads)
  - < 20°C → Scale Down (to save energy)

### 6. 🧮 Optimization with Linear Programming
- Two objectives:
  - Optimal cooling resource allocation
  - Efficient power consumption control
- Dynamically adjusts power output during environmental shifts

---

## 📊 Results

| Model              | MAE     | MSE     | R² Score |
|-------------------|---------|---------|----------|
| Random Forest      | 0.0579  | 0.0069  | 0.9930   |
| Linear Regression  | 0.0631  | 0.0071  | 0.9928   |
| Gradient Boosting  | 0.0644  | 0.0079  | 0.9920   |
| **XGBoost**        | **0.0520** | **0.0051** | **0.9947** |

✅ **Outcome**:
- Energy consumption significantly reduced
- Auto-scaling became proactive vs reactive
- Framework supports sustainable cloud operations in tropical regions

---

## 📈 Visual Architecture

```text
+---------------------------+
| NASA POWER Climate API   |
+------------+-------------+
             |
             v
+---------------------------+
| ML Models (XGBoost etc.) |
+------------+-------------+
             |
             v
+---------------------------+
| Temp Forecasts to S3     |
+------------+-------------+
             |
             v
+---------------------------+        +----------------------------+
| AWS Lambda Function      | -----> | CloudWatch Custom Metrics  |
+---------------------------+        +-------------+--------------+
                                                   |
                                        +----------v----------+
                                        | EC2 Auto Scaling     |
                                        | (Temp + CPU Metrics) |
                                        +----------------------+

```
---

## 🌱 Impact and Future Scope
Aligns with Green Computing and Sustainable Infrastructure

Future enhancements:

Include AQI, humidity, and rainfall

Use RNNs or LSTMs for advanced forecasting

Integrate renewable energy resource mapping

Expand to multi-region auto-scaling logic

## 👨‍💻 Team
Bandaru Jaya Nandini

Mamidi Leha Sahithi

Siddareddy Gari Harshika

Daggupati Praneesha

Ingeti Hemanth

Supervisor: Beena B. M. (ORCID: 0000-0001-9108-7073)

### Institution:
Amrita Vishwa Vidyapeetham, Amrita School of Computing, Bengaluru
