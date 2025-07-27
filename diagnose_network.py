#!/usr/bin/env python3
"""Diagnose network interfaces for NCCL on Snellius"""

import os
import socket
import subprocess
import sys

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "Command failed"

def diagnose_network():
    """Comprehensive network diagnosis"""
    print("=" * 80)
    print("NETWORK DIAGNOSIS FOR NCCL")
    print("=" * 80)
    
    # 1. Hostname and basic info
    print("\n1. BASIC INFORMATION:")
    print(f"Hostname: {socket.gethostname()}")
    print(f"SLURM Node: {os.environ.get('SLURMD_NODENAME', 'Not in SLURM')}")
    print(f"SLURM Job ID: {os.environ.get('SLURM_JOB_ID', 'Not in SLURM')}")
    
    # 2. Network interfaces
    print("\n2. NETWORK INTERFACES:")
    print(run_command("ip -o link show | awk '{print $2}' | sed 's/://g'"))
    
    print("\n3. INTERFACE DETAILS:")
    interfaces = run_command("ip -o link show | awk '{print $2}' | sed 's/://g'").split('\n')
    for iface in interfaces:
        if iface and iface != 'lo':
            print(f"\n{iface}:")
            print(run_command(f"ip addr show {iface} | grep inet"))
    
    # 3. InfiniBand check
    print("\n4. INFINIBAND STATUS:")
    ib_status = run_command("ibstat 2>/dev/null")
    if ib_status:
        print(ib_status)
    else:
        print("No InfiniBand detected")
    
    # 4. Environment variables
    print("\n5. NCCL ENVIRONMENT:")
    nccl_vars = [k for k in os.environ.keys() if 'NCCL' in k]
    for var in sorted(nccl_vars):
        print(f"{var}={os.environ[var]}")
    
    # 5. Recommendations
    print("\n6. RECOMMENDATIONS:")
    
    # Find best interface
    best_iface = None
    if 'ib0' in interfaces:
        best_iface = 'ib0'
        print("✓ InfiniBand (ib0) detected - best for NCCL")
    elif 'eth0' in interfaces:
        best_iface = 'eth0'
        print("✓ Ethernet (eth0) detected")
    else:
        # Find any ethernet interface
        for iface in interfaces:
            if iface.startswith('en') or iface.startswith('eth'):
                best_iface = iface
                print(f"✓ Found ethernet interface: {iface}")
                break
    
    if best_iface:
        print(f"\nRecommended NCCL settings:")
        print(f"export NCCL_SOCKET_IFNAME={best_iface}")
        print(f"export NCCL_IB_DISABLE=0")
        print(f"export NCCL_DEBUG=INFO")
    else:
        print("\n⚠️ No suitable network interface found!")
        print("Try: export NCCL_SOCKET_IFNAME=^lo")
    
    # 6. Test socket binding
    print("\n7. SOCKET BINDING TEST:")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
        s.close()
        print(f"✓ Can bind to port: {port}")
    except Exception as e:
        print(f"✗ Socket binding failed: {e}")

if __name__ == "__main__":
    diagnose_network()