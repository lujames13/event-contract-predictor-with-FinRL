"""
Simple script to ensure all necessary directories exist.
Run this before running the main application for the first time.
"""

import os

# Directories to ensure
directories = [
    "config",
    "data",
    "environments",
    "utils",
    "models",
    "logs",
    "cache",
    "cache/raw",
    "cache/processed",
    "cache/features"
]

# Create directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Ensured directory exists: {directory}")

print("Directory structure setup complete.")