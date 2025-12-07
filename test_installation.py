#!/usr/bin/env python3
"""
Installation Test Script for SVM Plant Classification

This script verifies that all dependencies are installed correctly
and the project structure is set up properly.

Run this after installation to ensure everything is working.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")

    packages = [
        ("sklearn", "scikit-learn"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("PIL", "Pillow"),
        ("cv2", "opencv-python"),
        ("skimage", "scikit-image"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("joblib", "joblib"),
        ("tqdm", "tqdm"),
    ]

    failed = []

    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError as e:
            print(f"  ✗ {package_name} - FAILED")
            failed.append(package_name)

    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Install with: pip install " + " ".join(failed))
        return False

    print("✓ All packages installed successfully!\n")
    return True


def test_project_structure():
    """Test that the project directory structure is correct."""
    print("Testing project structure...")

    required_dirs = [
        "src",
        "data",
        "data/train",
        "data/test",
        "models",
        "results",
    ]

    required_files = [
        "src/feature_extractor.py",
        "src/data_loader.py",
        "src/train_svm.py",
        "src/predict.py",
        "src/organize_dataset.py",
        "requirements.txt",
        "README.md",
    ]

    missing_dirs = []
    missing_files = []

    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"  ✓ {directory}/")
        else:
            print(f"  ✗ {directory}/ - MISSING")
            missing_dirs.append(directory)

    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            missing_files.append(file)

    if missing_dirs or missing_files:
        print(f"\n❌ Project structure incomplete")
        if missing_dirs:
            print(f"Missing directories: {', '.join(missing_dirs)}")
        if missing_files:
            print(f"Missing files: {', '.join(missing_files)}")
        return False

    print("✓ Project structure is correct!\n")
    return True


def test_module_imports():
    """Test that custom modules can be imported."""
    print("Testing custom module imports...")

    try:
        from feature_extractor import ImageFeatureExtractor

        print("  ✓ feature_extractor.ImageFeatureExtractor")
    except ImportError as e:
        print(f"  ✗ feature_extractor.ImageFeatureExtractor - {e}")
        return False

    try:
        from data_loader import PlantImageDataLoader

        print("  ✓ data_loader.PlantImageDataLoader")
    except ImportError as e:
        print(f"  ✗ data_loader.PlantImageDataLoader - {e}")
        return False

    try:
        from train_svm import SVMPlantClassifier

        print("  ✓ train_svm.SVMPlantClassifier")
    except ImportError as e:
        print(f"  ✗ train_svm.SVMPlantClassifier - {e}")
        return False

    print("✓ All custom modules imported successfully!\n")
    return True


def test_basic_functionality():
    """Test basic functionality with synthetic data."""
    print("Testing basic functionality...")

    try:
        import numpy as np
        from PIL import Image

        from feature_extractor import ImageFeatureExtractor

        # Create a test image
        print("  • Creating synthetic test image...")
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image_path = "test_image_temp.png"
        Image.fromarray(test_image).save(test_image_path)

        # Initialize feature extractor
        print("  • Initializing feature extractor...")
        extractor = ImageFeatureExtractor(
            image_size=(224, 224),
            use_color_histogram=True,
            use_hog=True,
            use_lbp=True,
            use_statistics=True,
        )

        # Extract features
        print("  • Extracting features from test image...")
        features = extractor.extract_features(test_image_path)

        print(f"  • Feature vector shape: {features.shape}")
        print(f"  • Feature dimension: {len(features)}")

        # Clean up
        os.remove(test_image_path)

        print("✓ Basic functionality test passed!\n")
        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}\n")
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        return False


def test_data_loader():
    """Test data loader functionality."""
    print("Testing data loader...")

    try:
        from data_loader import PlantImageDataLoader

        loader = PlantImageDataLoader(data_root="data")
        stats = loader.get_dataset_statistics()

        print(f"  • Training images: {stats['train']['total']}")
        print(f"  • Test images: {stats['test']['total']}")

        print("✓ Data loader test passed!\n")
        return True

    except Exception as e:
        print(f"❌ Data loader test failed: {e}\n")
        return False


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    print(f"  Python version: {version_str}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ✗ Python 3.8 or higher required")
        return False

    print(f"  ✓ Python version is compatible\n")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("SVM PLANT CLASSIFICATION - INSTALLATION TEST")
    print("=" * 70)
    print()

    tests = [
        ("Python Version", check_python_version),
        ("Package Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Custom Modules", test_module_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Data Loader", test_data_loader),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}\n")
            results.append((test_name, False))

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status:10} - {test_name}")

    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n🎉 SUCCESS! Installation is complete and working correctly.")
        print("\nNext steps:")
        print("  1. Read GETTING_STARTED.md for a quick tutorial")
        print("  2. Run: python quick_start.py")
        print(
            "  3. Or jump straight to: cd src && python train_svm.py --create-synthetic"
        )
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  • Install missing packages: pip install -r requirements.txt")
        print("  • Ensure you're in the svmTest directory")
        print("  • Check Python version (3.8+ required)")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
