from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from django.http import HttpResponse
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import numpy as np
import base64
import joblib
import os

# Used by all
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def home(request):
    return render(request, 'home.html')

def prediction(request):
    return render(request, 'prediction.html')
    
def update(request):
    return render(request, 'update.html')

def visualization(request):
    """ Uploads dataset and redirects to graph generation """
    if request.method == 'POST' and request.FILES.get('dataset'):
        file = request.FILES['dataset']
        
        # Save the file
        fs = FileSystemStorage(location=UPLOAD_FOLDER)
        filename = fs.save(file.name, file)
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        # Validate dataset
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                return render(request, 'visualization.html', {'error': 'Unsupported file format'})

            required_columns = {'Year', 'Month', 'Rain', 'Temparature'}
            if not required_columns.issubset(df.columns):
                return render(request, 'visualization.html', {'error': 'Dataset must contain Year, Month, Rain, and Temparature columns'})

            # Store the filename in session to access in the graph view
            request.session['uploaded_file'] = filename
            return redirect('generate_graphs')

        except Exception as e:
            return render(request, 'visualization.html', {'error': f'Error processing file: {e}'})

    return render(request, 'visualization.html')

### ** Generate Graphs and Display Them**
def generate_graphs(request):
    """ Generates multiple graphs from uploaded dataset """
    filename = request.session.get('uploaded_file')
    if not filename:
        return render(request, 'visualization.html', {'error': 'No file uploaded'})

    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # Load dataset
    if filename.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # Generate graphs
    graphs = {}

    def save_plot(fig):
        """ Convert Matplotlib figure to base64 PNG """
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    # Rainfall vs Temperature Scatter Plot
    fig, ax = plt.subplots()
    ax.scatter(df['Rain'], df['Temparature'], color='blue', alpha=0.5)
    ax.set_xlabel('Rainfall (mm)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Rainfall vs Temperature')
    graphs['rain_temp'] = save_plot(fig)
    plt.close(fig)

    # Monthly Rainfall Bar Chart
    fig, ax = plt.subplots()
    df.groupby('Month')['Rain'].mean().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Average Monthly Rainfall')
    graphs['monthly_rain'] = save_plot(fig)
    plt.close(fig)

    # Yearly Average Temperature Line Chart
    fig, ax = plt.subplots()
    df.groupby('Year')['Temparature'].mean().plot(ax=ax, marker='o', linestyle='-')
    ax.set_title('Yearly Average Temperature')
    graphs['yearly_temp'] = save_plot(fig)
    plt.close(fig)

    return render(request, 'graphs.html', {'graphs': graphs})


def analysis(request):
    return render(request, 'analysis.html')

def process_weather_analysis(request):
    if request.method == 'POST' and request.FILES['dataset']:
        file = request.FILES['dataset']

        fs = FileSystemStorage(location=UPLOAD_FOLDER)
        filename = fs.save(file.name, file)
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        try:
            # Read the dataset
            if file.name.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                return render(request, 'analysis.html', {'error': 'Unsupported file format'})

            # Check for necessary columns in dataset
            required_columns = {'Year', 'Month', 'Rain', 'Temparature'}
            if not required_columns.issubset(df.columns):
                return render(request, 'analysis.html', {'error': 'Dataset must contain Year, Month, Rain, and Temparature columns'})

            # Perform analysis here (e.g., linear regression, KMeans, etc.)
            temp_seasonal_analysis = analyze_seasonal_temp(df)
            rain_seasonal_analysis = analyze_seasonal_rain(df)
            regression_analysis = linear_regression_analysis(df)
            yearly_trends = analyze_yearly_trends(df)
            month_vs_temp_rain = analyze_month_vs_temp_rain(df)

            # Pass the analysis to the result page
            return render(request, 'weather_analysis_result.html', {
                'temp_seasonal_analysis': temp_seasonal_analysis,
                'rain_seasonal_analysis': rain_seasonal_analysis,
                'regression_analysis': regression_analysis,
                'yearly_trends': yearly_trends,
                'month_vs_temp_rain': month_vs_temp_rain
            })
        except Exception as e:
            return render(request, 'analysis.html', {'error': f'Error processing file: {e}'})
    return render(request, 'analysis.html')

def analyze_seasonal_temp(df):
    """Analyze seasonal temperature trends"""
    seasonal_temp = df.groupby('Month')['Temparature'].mean().sort_values()
    return seasonal_temp

def analyze_seasonal_rain(df):
    """Analyze seasonal rainfall trends"""
    seasonal_rain = df.groupby('Month')['Rain'].mean().sort_values()
    return seasonal_rain

def linear_regression_analysis(df):
    """Perform linear regression between Year and Rain"""
    X = df[['Year']]
    y = df['Rain']
    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict([[2025]])  # Predict for the next year
    return f"Predicted rainfall for 2025: {prediction[0]:.2f} mm"

def analyze_yearly_trends(df):
    """Analyze yearly trends in temperature and rainfall"""
    yearly_temp = df.groupby('Year')['Temparature'].mean()
    yearly_rain = df.groupby('Year')['Rain'].mean()
    return f"Yearly Temperature Trends: {yearly_temp}, Yearly Rainfall Trends: {yearly_rain}"

def analyze_month_vs_temp_rain(df):
    """Analyze the relationship between Month, Temperature, and Rainfall"""
    correlation = df[['Month', 'Temparature', 'Rain']].corr()
    return f"Correlation between Month, Temperature, and Rainfall: {correlation}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
temp_model_path = os.path.join(BASE_DIR, 'static/model/temp_model.joblib')
rain_model_path = os.path.join(BASE_DIR, 'static/model/rain_model.joblib')

temp_model = joblib.load(temp_model_path)
rain_model = joblib.load(rain_model_path)

def predict_weather(request):
    if request.method == 'POST':
        month = int(request.POST['month'])
        year = int(request.POST['year'])
        prediction_type = request.POST.get('predictionType', 'rainfall')

        if prediction_type == 'rainfall':
            temperature = float(request.POST['temperature'])
            input_data = np.array([[year, month, temperature]])  # Format for model
            predicted_rainfall = rain_model.predict(input_data)[0]
            result = f"Predicted Rainfall for {month}/{year} with {temperature}°C is {predicted_rainfall:.2f} mm."
        
        else:  # Predict temperature
            rainfall = float(request.POST['rainfall'])
            input_data = np.array([[year, month, rainfall]])
            predicted_temperature = temp_model.predict(input_data)[0]
            result = f"Predicted Temperature for {month}/{year} with {rainfall} mm rain is {predicted_temperature:.2f}°C."

        return render(request, 'weather_prediction_result.html', {'result': result})

    return render(request, 'weather_prediction.html')