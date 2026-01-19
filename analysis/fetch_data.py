import pandas as pd
import numpy as np
import requests
import io

# URL for a sample of MIT-BIH Arrhythmia Dataset (Lead II)
# Hosted on a stable public mirror for educational use
DATA_URL = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/ecg.csv"

def get_ecg_data():
    print(f"Downloading ECG data from {DATA_URL}...")
    s = requests.get(DATA_URL).content
    df = pd.read_csv(io.StringIO(s.decode('utf-8')))
    
    # The dataset usually has columns like 'ecg_value'. 
    # This specific file has no headers, just values.
    data = df.iloc[:, 0].values
    
    # Normalize data (Industry Practice: Mean 0, Std 1)
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std
    
    np.savetxt("ecg_data.txt", data, fmt='%.6f')
    print(f"Saved {len(data)} data points to ecg_data.txt")

if __name__ == "__main__":
    try:
        get_ecg_data()
    except Exception as e:
        print(f"Error: {e}")
        print("Please install pandas/requests: pip install pandas requests")
