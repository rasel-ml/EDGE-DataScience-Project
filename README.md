# Weather Prediction and Analysis Web App

## Overview
This project is a Django-based web application that allows users to:
- **Predict weather conditions** (rainfall or temperature) based on historical data.
- **Visualize weather trends** using uploaded datasets.
- **Analyze weather patterns** using machine learning techniques.
- **Update weather-related datasets.**

The project was developed as part of the **Data Science with Python** course and utilizes **scikit-learn**, **Matplotlib**, **Seaborn**, and **Pandas** for data analysis and visualization.

## Features
### 1. Weather Prediction
- Users can input **Month, Year, and Temperature** to predict **Rainfall**.
- Alternatively, users can input **Month, Year, and Rainfall** to predict **Temperature**.
- The model is trained once using historical weather data from **1901 to 2023**.
- Uses **Linear Regression** from **scikit-learn**.

### 2. Weather Visualization
- Users can upload a dataset (CSV, XLSX) containing columns: **Year, Month, Rain, Temperature**.
- The application generates various graphs like:
  - Rainfall vs. Temperature
  - Monthly Rainfall Trends
  - Yearly Temperature Changes
  - Other statistical insights.

### 3. Weather Analysis
- Uses **Linear Regression and K-Means Clustering** to analyze patterns such as:
  - **Which months have the highest/lowest rainfall?**
  - **Which years had extreme weather conditions?**
  - **Temperature trends over the years.**
  - **Seasonal variations in rainfall and temperature.**

## Technologies Used
### Programming Languages & Frameworks
- **[Python](https://www.python.org/)** (Backend logic, Machine Learning, Data Processing)
- **[Django](https://www.djangoproject.com/)** (Web Framework)
- **[HTML](https://html.com/), [CSS](https://www.w3.org/Style/CSS/Overview.en.html), [JavaScript](https://www.javascript.com/)** (Frontend)
- **[Jupyter Notebook](https://jupyter.org/)** (Model training & Data analysis)

### Libraries & Tools
- **[Pandas](https://pandas.pydata.org/)** (Data handling)
- **[Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)** (Visualization)
- **[scikit-learn](https://scikit-learn.org/)** (Machine Learning models: Linear Regression, K-Means Clustering)
- **[Joblib](https://joblib.readthedocs.io/en/stable/)** (Model persistence for predictions)

## Installation & Setup
### Prerequisites
Ensure you have **Python 3.7+** and **pip** installed. Then, install the required dependencies:
```sh
pip install django pandas matplotlib seaborn scikit-learn joblib git numpy
```

### Run the Project
To run this project, open the command prompt and follow the steps below...
1. Clone this repository:
```sh
git clone https://github.com/rasel-ml/EDGE-DataScience-Project.git
cd EDGE-DataScience-Project
```
2. Run Django migrations:
```sh
python manage.py migrate
```
3. Start the development server:
```sh
python manage.py runserver
```
4. Open your browser and go to:
```
http://127.0.0.1:8000/
```

## Project Structure
Weather Prediction and Analysis Web App
```
├── myapp/
│   ├── migrations/
│   │   └── __init__.py
│   ├── static/
│   │   ├── dataset
│   │   │   └── Weather_data.csv
│   │   ├── image/
│   │   │   ├── analysis.jpeg
│   │   │   ├── bg.jpeg
│   │   │   ├── prediction.jpeg
│   │   │   ├── update.jpeg
│   │   │   └── visualization.jpeg
│   │   ├── model/
│   │   │   ├── rain_model.joblib
│   │   │   └── temp_model.joblib
│   │   ├── script/
│   │   │   └── prediction.js
│   │   └── style/
│   │       ├── graphs.css
│   │       ├── home.css
│   │       ├── prediction.css
│   │       └── visualization.css
│   ├── templates/
│   │   ├── analysis.html
│   │   ├── graphs.html
│   │   ├── home.html
│   │   ├── prediction.html
│   │   ├── process.html
│   │   ├── update.html
│   │   ├── visualization.html
│   │   ├── weather_analysis_result.html
│   │   └── weather_prediction_result.html
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── mysite/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── Project Report/
│   └── Data Science Project Report on Weather Prediction and Analysis.pdf
│── uploads/
│── db.sqlite3
│── manage.py
│── requirements.txt
└── README.md
```

## How to Use the App

### 1. Weather Analysis
- Navigate to **Weather Analysis**.
- Upload a dataset.
- Get insights using **Regression and Clustering models**.

### 2. Weather Visualization
- Go to **Weather Visualization**.
- Upload a dataset (CSV or XLSX).
- Click **Upload** to generate graphs.

### 3. Weather Prediction
- Navigate to **Weather Prediction** page.
- Choose between **Rainfall Prediction** or **Temperature Prediction**.
- Enter the required input values.
- Click **Predict** to get results.

## Dataset
The dataset used for training and analysis was collected from **[Kaggle](https://www.kaggle.com/datasets/yakinrubaiat/bangladesh-weather-dataset)**, containing historical weather data from **1901 to 2023**.

## License
This project is open-source and available under the **[MIT License](https://opensource.org/license/mit)**.

## Author
**[Md. Rasel Molla](https://www.linkedin.com/in/rasel-molla-9a6597347)**
