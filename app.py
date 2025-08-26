from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import os

app = Flask(__name__)

# Get the absolute path of the directory containing the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the full paths to the model files
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
model_path = os.path.join(BASE_DIR, 'models', 'customer_segmentation.pkl')

# Create models directory if it doesn't exist
models_dir = os.path.join(BASE_DIR, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Load the saved models or create new ones
try:
    scaler = pickle.load(open(scaler_path, 'rb'))
    model = pickle.load(open(model_path, 'rb'))
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Creating and saving new models...")
    
    # Import required modules
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    # Create sample data for training (using your provided dataset)
    csv_data = """
CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
4,Female,23,16,77
5,Female,31,17,40
6,Female,22,17,76
7,Female,35,18,6
8,Female,23,18,94
9,Male,64,19,3
10,Female,30,19,72
11,Male,67,19,14
12,Female,35,19,99
13,Female,58,20,15
14,Female,24,20,77
15,Male,37,20,13
16,Male,22,20,79
17,Female,35,21,35
18,Male,20,21,66
19,Female,52,23,29
20,Female,35,23,98
21,Male,35,24,35
22,Male,25,24,73
23,Female,46,25,5
24,Male,31,25,73
25,Female,54,28,14
26,Female,29,28,82
27,Male,45,28,32
28,Female,35,28,61
29,Male,40,29,31
30,Female,23,29,87
31,Male,60,30,4
32,Female,21,30,73
33,Female,53,33,4
34,Male,18,33,92
35,Female,49,33,14
36,Female,21,33,81
37,Male,42,34,17
38,Female,30,34,73
39,Female,36,37,26
40,Female,20,37,75
41,Male,65,38,35
42,Female,24,38,92
43,Male,48,39,36
44,Male,31,39,61
45,Female,49,39,28
46,Male,24,39,65
47,Female,50,40,55
48,Female,27,40,47
49,Male,29,40,42
50,Female,31,40,42
"""
    from io import StringIO
    df = pd.read_csv(StringIO(csv_data))
    
    # Select features for clustering
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train K-means model
    model = KMeans(n_clusters=4, random_state=42)
    model.fit(X_scaled)
    
    # Save the models
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print("New models created and saved successfully!")

# Create static directory if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = int(request.form['age'])
        income = int(request.form['income'])
        spending = int(request.form['spending'])
        
        # Create feature array
        features = np.array([[age, income, spending]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict cluster
        cluster = model.predict(features_scaled)[0]
        
        # Get cluster centers
        centers = scaler.inverse_transform(model.cluster_centers_)
        
        # Define cluster names and descriptions
        cluster_names = [
            "Balanced Customers",
            "High Income Low Spenders",
            "Low Income High Spenders",
            "Premium Customers"
        ]
        
        cluster_descriptions = [
            "Middle-aged customers with moderate income and balanced spending.",
            "Older customers with high income but conservative spending habits.",
            "Younger customers with lower income but high spending tendencies.",
            "Affluent customers with high income and high spending patterns."
        ]
        
        # Create visualization
        plt.figure(figsize=(10, 6), facecolor='#1e1e1e')
        ax = plt.gca()
        ax.set_facecolor('#2c3e50')
        
        # Plot all cluster centers
        ax.scatter(centers[:, 1], centers[:, 2], c='black', s=200, marker='X', label='Cluster Centers')
        
        # Plot the customer point
        ax.scatter(income, spending, c='#3498db', s=100, marker='o', label='Customer')
        
        # Add labels and title
        ax.set_xlabel('Annual Income (k$)', color='#e0e0e0')
        ax.set_ylabel('Spending Score (1-100)', color='#e0e0e0')
        ax.set_title(f'Customer Segmentation - Cluster {cluster}', color='#ffffff')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Change tick colors
        ax.tick_params(axis='x', colors='#e0e0e0')
        ax.tick_params(axis='y', colors='#e0e0e0')
        
        # Save plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png', facecolor='#1e1e1e', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()
        
        # Prepare cluster information
        cluster_info = {
            'cluster': int(cluster),
            'cluster_name': cluster_names[cluster],
            'cluster_description': cluster_descriptions[cluster],
            'age': age,
            'income': income,
            'spending': spending,
            'plot_url': plot_url,
            'centers': centers.tolist()
        }
        
        return jsonify(cluster_info)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
