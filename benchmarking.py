import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
import sys
import os

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Conv1 -> Chomp -> ReLU -> Dropout
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Conv2 -> Chomp -> ReLU -> Dropout
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class PyTorchTCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(PyTorchTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # Padding such that output length = input length
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        self.final = nn.Conv1d(num_channels[-1], 1, 1)

    def forward(self, x):
        y = self.network(x)
        return self.final(y)

def benchmark_pytorch(device_name):
    device = torch.device(device_name)
    if device_name == 'cuda' and not torch.cuda.is_available():
        print(" [SKIP] CUDA not available for PyTorch.")
        return None

    print(f"--- Benchmarking PyTorch [{device_name.upper()}] ---")
    
    model = PyTorchTCN(1, [16, 16, 16, 16], kernel_size=3, dropout=0.0).to(device)
    model.eval() # Inference mode
    
    x = torch.randn(1, 1, 2000).to(device)
    
    for _ in range(10):
        _ = model(x)
    
    iters = 1000
    if device_name == 'cuda': torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
            
    if device_name == 'cuda': torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_latency = ((end - start) / iters) * 1000.0 # ms
    print(f" Average Inference Time: {avg_latency:.4f} ms")
    return avg_latency

def check_cpp_output(filename="final_test_results.csv"):
    print(f"\n--- Checking C++ Output Quality ({filename}) ---")
    if not os.path.exists(filename):
        print("❌ File not found. Did C++ run?")
        return
    
    try:
        df = pd.read_csv(filename)
        preds = df['Predicted'].values
        
        if np.isnan(preds).any():
            print("❌ CRITICAL: C++ Output contains NaNs!")
            return

        if np.all(preds == 0):
            print("❌ CRITICAL: C++ Output is all Zeros!")
            return

        std = np.std(preds)
        print(f" Prediction Std Dev: {std:.4f}")
        if std < 0.001:
            print("⚠️  WARNING: Low variance. Model might be stuck at mean.")
        else:
            print("✅ Output looks healthy (non-trivial variance).")
            
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")

if __name__ == "__main__":
    py_cpu = benchmark_pytorch('cpu')
    py_gpu = benchmark_pytorch('cuda')
    
    check_cpp_output("final_test_results.csv")
    
    print("\n" + "="*40)
    print("   THE GRAND PRIX RESULTS (Inference)")
    print("="*40)
    print(f"{'Engine':<20} | {'Device':<5} | {'Time (ms)':<10}")
    print("-" * 40)
    
    if py_cpu: print(f"{'PyTorch':<20} | {'CPU':<5} | {py_cpu:.4f} ms")
    if py_gpu: print(f"{'PyTorch':<20} | {'GPU':<5} | {py_gpu:.4f} ms")
    
    print("-" * 40)
    print("NOTE: Compare these PyTorch numbers with the 'Time' output")
    print("from the C++ executable (divide C++ total time by iterations")
    print("if C++ reported total training time).")
    print("="*40)
