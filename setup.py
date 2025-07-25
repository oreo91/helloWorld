#!/usr/bin/env python3
"""
Setup script for 3D Model Builder from Images
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies."""
    print("🔧 Installing dependencies...")
    
    # Install basic requirements
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Basic dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install basic dependencies: {e}")
        return False
    
    # Try to install PyTorch with CUDA support if available
    try:
        print("🚀 Attempting to install PyTorch with CUDA support...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
        print("✅ PyTorch with CUDA installed successfully")
    except subprocess.CalledProcessError:
        print("⚠️  CUDA installation failed, using CPU-only PyTorch")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio"
            ])
            print("✅ PyTorch (CPU) installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install PyTorch: {e}")
    
    return True

def check_installation():
    """Check if installation was successful."""
    print("🔍 Checking installation...")
    
    required_packages = [
        'cv2', 'numpy', 'open3d', 'matplotlib', 
        'sklearn', 'PIL', 'gradio'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {missing}")
        return False
    else:
        print("✅ All core packages installed successfully")
        return True

def create_test_directories():
    """Create necessary directories for testing."""
    print("📁 Creating test directories...")
    
    directories = [
        "sample_images",
        "output_3d",
        "demo_output_traditional",
        "demo_output_neural",
        "demo_output_comparison"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}/")
    
    print("✅ Test directories created")

def main():
    """Main setup function."""
    print("🏗️  3D MODEL BUILDER FROM IMAGES - SETUP")
    print("="*50)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        return 1
    
    # Check installation
    if not check_installation():
        print("❌ Setup failed during installation check")
        return 1
    
    # Create directories
    create_test_directories()
    
    print("\n" + "="*50)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("You can now use the 3D Model Builder:")
    print()
    print("Web Interface:")
    print("  python web_interface.py")
    print()
    print("Command Line:")
    print("  python cli_interface.py --help")
    print()
    print("Demo:")
    print("  python demo.py")
    print()
    print("Python API:")
    print("  from model_builder_3d import Model3DBuilder")
    print("="*50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())