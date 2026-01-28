import json
from datetime import datetime

def load_lottie_file(filepath: str):
    """Load a Lottie animation file and return its content."""
    with open(filepath, "r") as f:
        return json.load(f)

def safe_parse_datetime(timestamp_str):
    """Parse datetime strings safely, handling both old and new formats"""
    if isinstance(timestamp_str, str):
        formats = [
            "%d/%m/%Y %H:%M:%S",  # New format
            "%Y-%m-%d %H:%M:%S",   # Old format
            "%d/%m/%Y %H:%M",      # New format without seconds
            "%Y-%m-%d %H:%M",      # Old format without seconds
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        # If all parsing attempts fail, return current time
        print(f"Warning: Could not parse datetime string: {timestamp_str}")
        return datetime.now()
    return timestamp_str  # If it's already a datetime object 