"""
Fallback implementation for psutil functionality
Provides minimal system monitoring when psutil is unavailable
"""
import time
from typing import Dict, Any

class Process:
    def __init__(self, pid=None):
        self.pid = pid or 1
        
    def memory_info(self):
        class MemInfo:
            rss = 50 * 1024 * 1024  # 50MB fallback
            vms = 100 * 1024 * 1024  # 100MB fallback
        return MemInfo()
    
    def cpu_percent(self):
        return 0.0

def virtual_memory():
    class VMemInfo:
        total = 8 * 1024 * 1024 * 1024  # 8GB fallback
        available = 4 * 1024 * 1024 * 1024  # 4GB fallback
        percent = 50.0
    return VMemInfo()

def cpu_percent():
    return 0.0

def disk_usage(path):
    class DiskInfo:
        total = 100 * 1024 * 1024 * 1024  # 100GB fallback
        used = 50 * 1024 * 1024 * 1024   # 50GB fallback  
        free = 50 * 1024 * 1024 * 1024   # 50GB fallback
    return DiskInfo()
