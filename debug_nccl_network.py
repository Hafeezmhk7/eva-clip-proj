#!/usr/bin/env python3
"""Debug NCCL network issues on Snellius"""

import os
import socket
import subprocess
import sys
import torch

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "Failed"

def debug_network():
    print("=" * 80)
    print("NCCL NETWORK DEBUGGING")
    print("=" * 80)
    
    # 1. Environment
    print("\n1. ENVIRONMENT VARIABLES:")
    for var in ["NCCL_SOCKET_IFNAME", "NCCL_IB_DISABLE", "NCCL_DEBUG", 
                "MASTER_ADDR", "MASTER_PORT", "SLURM_NODELIST"]:
        print(f"{var}: {os.environ.get(var, 'not set')}")
    
    # 2. Network interfaces with IPs
    print("\n2. NETWORK INTERFACES WITH IPs:")
    interfaces_output = run_cmd("ip -4 addr show")
    print(interfaces_output)
    
    # 3. Check if interfaces are UP
    print("\n3. INTERFACE STATUS:")
    for iface in ["ib0", "team-clmgt", "ens4f0", "ens4f1"]:
        status = run_cmd(f"ip link show {iface} 2>/dev/null | grep -o 'state [^ ]*'")
        print(f"{iface}: {status}")
    
    # 4. Test socket binding on each interface
    print("\n4. SOCKET BINDING TEST PER INTERFACE:")
    interfaces = {
        "ib0": "172.22.63.191",
        "team-clmgt": "172.18.63.191",
        "localhost": "127.0.0.1"
    }
    
    for iface, ip in interfaces.items():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((ip, 0))
            port = s.getsockname()[1]
            s.close()
            print(f"{iface} ({ip}): ✓ Can bind to port {port}")
        except Exception as e:
            print(f"{iface} ({ip}): ✗ {e}")
    
    # 5. PyTorch CUDA info
    print("\n5. PYTORCH/CUDA INFO:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"NCCL version: {torch.cuda.nccl.version()}")

if __name__ == "__main__":
    debug_network()
    