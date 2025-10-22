#!/usr/bin/env python3
"""
Basic test script to verify the 3D Model Builder structure
"""

import sys
import os

def test_file_structure():
    """Test that all required files are present."""
    print("🔍 Testing file structure...")
    
    required_files = [
        "model_builder_3d.py",
        "neural_reconstruction.py", 
        "web_interface.py",
        "cli_interface.py",
        "demo.py",
        "setup.py",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"  ✅ {file}")
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def test_imports():
    """Test basic Python imports."""
    print("\n🔍 Testing basic imports...")
    
    # Test built-in modules
    try:
        import os
        import sys
        import json
        import logging
        print("  ✅ Built-in modules")
    except ImportError as e:
        print(f"  ❌ Built-in modules failed: {e}")
        return False
    
    # Test if files can be imported (syntax check)
    sys.path.insert(0, os.getcwd())
    
    try:
        # Just check syntax, don't actually import (dependencies may be missing)
        import ast
        
        files_to_check = [
            "model_builder_3d.py",
            "neural_reconstruction.py",
            "web_interface.py", 
            "cli_interface.py",
            "demo.py"
        ]
        
        for file in files_to_check:
            with open(file, 'r') as f:
                source = f.read()
            
            try:
                ast.parse(source)
                print(f"  ✅ {file} (syntax valid)")
            except SyntaxError as e:
                print(f"  ❌ {file} (syntax error): {e}")
                return False
        
    except Exception as e:
        print(f"  ❌ Import test failed: {e}")
        return False
    
    print("✅ All Python files have valid syntax")
    return True

def test_requirements():
    """Test requirements.txt format."""
    print("\n🔍 Testing requirements.txt...")
    
    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
        
        packages = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                packages.append(line.split('==')[0])
        
        print(f"  📦 Found {len(packages)} packages:")
        for pkg in packages[:5]:  # Show first 5
            print(f"    - {pkg}")
        if len(packages) > 5:
            print(f"    ... and {len(packages) - 5} more")
        
        print("✅ Requirements file is valid")
        return True
        
    except Exception as e:
        print(f"  ❌ Requirements test failed: {e}")
        return False

def test_documentation():
    """Test documentation completeness."""
    print("\n🔍 Testing documentation...")
    
    try:
        with open('README.md', 'r') as f:
            content = f.read()
        
        required_sections = [
            "# 🏗️ 3D Model Builder",
            "## ✨ Features", 
            "## 🚀 Quick Start",
            "## 📖 Detailed Usage"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"  ❌ Missing sections: {missing_sections}")
            return False
        
        print(f"  ✅ Documentation has {len(content)} characters")
        print("✅ Documentation is complete")
        return True
        
    except Exception as e:
        print(f"  ❌ Documentation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🏗️  3D MODEL BUILDER - BASIC TESTS")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_imports,
        test_requirements,
        test_documentation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("The 3D Model Builder structure is ready!")
        print("\nNext steps:")
        print("1. Install dependencies: python3 setup.py")
        print("2. Run demo: python3 demo.py")
        print("3. Start web interface: python3 web_interface.py")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())