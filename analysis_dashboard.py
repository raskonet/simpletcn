import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os

# --- Configuration for "Pitch Deck" Aesthetics ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Colors: Professional Medical Tech Palette
COLOR_ACTUAL = '#2C3E50'  # Deep Blue/Slate
COLOR_PRED = '#E74C3C'    # Urgent Red
COLOR_RESID = '#8E44AD'   # Purple

def load_data(filename="final_test_results.csv"):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Run ./tcn_app first.")
        sys.exit(1)
    
    print(f"Loading {filename}...")
    df = pd.read_csv(filename)
    return df

def generate_executive_summary(df):
    y_true = df['Actual']
    y_pred = df['Predicted']
    
    # Metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson = np.corrcoef(y_true, y_pred)[0, 1]
    
    print("\n" + "="*40)
    print("   MODEL PERFORMANCE EXECUTIVE SUMMARY")
    print("="*40)
    print(f"{'Metric':<25} | {'Value':<10}")
    print("-" * 40)
    print(f"{'R-Squared (Fit Quality)':<25} | {r2:.4f}")
    print(f"{'Pearson Correlation':<25} | {pearson:.4f}")
    print(f"{'RMSE (Root Mean Sq Error)':<25} | {rmse:.4f}")
    print(f"{'MAE (Mean Abs Error)':<25} | {mae:.4f}")
    print("-" * 40)
    
    if r2 > 0.8:
        print(">> VERDICT: EXCELLENT Predictive Power")
    elif r2 > 0.5:
        print(">> VERDICT: MODERATE Predictive Power")
    else:
        print(">> VERDICT: NEEDS IMPROVEMENT")
    print("="*40 + "\n")

def plot_macro_timeline(df, filename="viz_01_macro_timeline.png"):
    """Slide 1: The Big Picture. How does the model track over time?"""
    plt.figure(figsize=(12, 5))
    
    # Plot a subset to avoid rendering artifact clutter (last 2000 points)
    subset = df.tail(2000)
    
    plt.plot(subset.index, subset['Actual'], label='Ground Truth (ECG)', 
             color=COLOR_ACTUAL, linewidth=1.5, alpha=0.8)
    plt.plot(subset.index, subset['Predicted'], label='TCN Forecast (Next Step)', 
             color=COLOR_PRED, linewidth=1, linestyle='--', alpha=0.9)
    
    plt.title("Macro View: Forecasting Performance (Last 2000 Time Steps)", fontsize=14, fontweight='bold')
    plt.ylabel("Normalized Amplitude (Z-Score)")
    plt.xlabel("Time Step Index")
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"[Saved] {filename}")
    plt.close()

def plot_micro_zoom(df, filename="viz_02_micro_zoom.png"):
    """Slide 2: The Detail View. Can it capture the QRS complex peaks?"""
    plt.figure(figsize=(10, 6))
    
    # Zoom into a specific window with distinct features
    # Assuming standard ECG, 200 points is about 2 seconds
    start_idx = len(df) - 300
    end_idx = len(df) - 100
    subset = df.iloc[start_idx:end_idx]
    
    plt.plot(subset.index, subset['Actual'], label='Actual', color=COLOR_ACTUAL, linewidth=2.5)
    plt.plot(subset.index, subset['Predicted'], label='Predicted', color=COLOR_PRED, 
             linewidth=2, linestyle=':')
    
    # Fill area between curves to highlight error
    plt.fill_between(subset.index, subset['Actual'], subset['Predicted'], 
                     color='gray', alpha=0.1, label='Error Margin')

    plt.title("Micro View: Signal Fidelity Analysis", fontsize=14, fontweight='bold')
    plt.suptitle("Detailed tracking of P-QRS-T complexes", fontsize=10, y=0.93)
    plt.ylabel("Amplitude")
    plt.xlabel("Time Step")
    plt.legend(loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"[Saved] {filename}")
    plt.close()

def plot_regression_scatter(df, filename="viz_03_correlation.png"):
    """Slide 3: Goodness of Fit. Ideally a 45-degree line."""
    plt.figure(figsize=(8, 8))
    
    # Sample down for scatter plot performance if dataset is huge
    sample = df.sample(n=min(5000, len(df)), random_state=42)
    
    sns.scatterplot(x=sample['Actual'], y=sample['Predicted'], 
                    alpha=0.3, color='#2980B9', edgecolor=None)
    
    # Perfect fit line
    min_val = min(sample['Actual'].min(), sample['Predicted'].min())
    max_val = max(sample['Actual'].max(), sample['Predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1.5, label='Ideal Fit (y=x)')
    
    r2 = r2_score(df['Actual'], df['Predicted'])
    plt.text(min_val + 0.5, max_val - 1.0, f"$R^2 = {r2:.3f}$", fontsize=16, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8))

    plt.title("Prediction Correlation Analysis", fontsize=14, fontweight='bold')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"[Saved] {filename}")
    plt.close()

def plot_error_distribution(df, filename="viz_04_error_distribution.png"):
    """Slide 4: Error Bias. Is the error centered at zero?"""
    plt.figure(figsize=(10, 6))
    
    residuals = df['Actual'] - df['Predicted']
    
    sns.histplot(residuals, bins=100, kde=True, color=COLOR_RESID, line_kws={'linewidth': 2})
    
    mean_err = residuals.mean()
    std_err = residuals.std()
    
    plt.axvline(mean_err, color='black', linestyle='--', label=f'Mean Bias: {mean_err:.4f}')
    
    plt.title("Residual Error Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"[Saved] {filename}")
    plt.close()

def plot_dashboard(df, filename="viz_00_dashboard_summary.png"):
    """Slide 0: One-Pager Dashboard combining everything."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Timeline
    ax1 = fig.add_subplot(gs[0, :])
    subset = df.tail(500)
    ax1.plot(subset.index, subset['Actual'], color=COLOR_ACTUAL, label='Actual')
    ax1.plot(subset.index, subset['Predicted'], color=COLOR_PRED, linestyle='--', label='Pred')
    ax1.set_title("Forecast Timeline (Last 500 points)", fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter
    ax2 = fig.add_subplot(gs[1, 0])
    sample = df.sample(n=min(2000, len(df)))
    ax2.scatter(sample['Actual'], sample['Predicted'], alpha=0.2, color='#2980B9', s=10)
    min_v, max_v = sample['Actual'].min(), sample['Actual'].max()
    ax2.plot([min_v, max_v], [min_v, max_v], 'k--')
    ax2.set_title("Correlation Check")
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    
    # 3. Residuals
    ax3 = fig.add_subplot(gs[1, 1])
    residuals = df['Actual'] - df['Predicted']
    sns.kdeplot(residuals, fill=True, color=COLOR_RESID, ax=ax3)
    ax3.set_title("Error Density (Bias Check)")
    ax3.set_xlabel("Error Magnitude")
    
    plt.suptitle("TCN Model Performance Dashboard", fontsize=20, y=0.98)
    plt.savefig(filename)
    print(f"[Saved] {filename}")
    plt.close()

if __name__ == "__main__":
    # 1. Load Data
    data = load_data()
    
    # 2. Text Summary
    generate_executive_summary(data)
    
    # 3. Generate Visual Assets
    print("Generating Visualizations...")
    plot_dashboard(data)
    plot_macro_timeline(data)
    plot_micro_zoom(data)
    plot_regression_scatter(data)
    plot_error_distribution(data)
    
    print("\n[DONE] All assets generated.")
