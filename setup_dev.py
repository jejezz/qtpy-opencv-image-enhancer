#!/usr/bin/env python3
"""
Development Setup Script
Run this to set up the development environment
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False

def main():
    """Set up the development environment."""
    print("🚀 Setting up Qt Image Enhancer development environment")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("⚠️  Warning: Not in a virtual environment")
        print("   It's recommended to use a virtual environment:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print()
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("Setup failed. Please check the error messages above.")
        return False
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    try:
        import qtpy
        from qtpy.QtWidgets import QApplication
        print("✅ QtPy installation verified")
        
        import PIL
        print("✅ Pillow installation verified")
        
        import numpy
        print("✅ NumPy installation verified")
        
        import cv2
        print(f"✅ OpenCV installation verified (version: {cv2.__version__})")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo run the application:")
    print("   python main.py")
    print("\nHappy coding! 🖼️✨")
    
    return True

if __name__ == "__main__":
    main()