# Installation Guide

Complete installation instructions for the SVM Plant Classification project.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Install](#quick-install)
- [Detailed Installation](#detailed-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Platform-Specific Notes](#platform-specific-notes)

---

## Prerequisites

Before installing, ensure you have:

- **Python 3.8 or higher** ([Download Python](https://www.python.org/downloads/))
- **pip** package manager (usually included with Python)
- **4GB+ RAM** (recommended for training)
- **500MB+ free disk space**

### Check Your Python Version

```bash
python --version
# or
python3 --version
```

You should see something like `Python 3.8.x` or higher.

### Check pip

```bash
pip --version
# or
pip3 --version
```

---

## Quick Install

For experienced users, here's the quick version:

```bash
# Clone or navigate to project
cd svmTest

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_installation.py

# Run quick start
python quick_start.py
```

---

## Detailed Installation

### Step 1: Set Up Directory

Navigate to the project directory:

```bash
cd svmTest
```

Verify you're in the correct location:

```bash
ls
# You should see: README.md, requirements.txt, src/, data/, etc.
```

### Step 2: Create Virtual Environment (Recommended)

A virtual environment keeps your project dependencies isolated.

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your command prompt.

**To deactivate later:**
```bash
deactivate
```

### Step 3: Upgrade pip (Optional but Recommended)

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- scikit-learn (Machine Learning)
- numpy (Numerical computing)
- scipy (Scientific computing)
- Pillow (Image processing)
- opencv-python (Computer vision)
- scikit-image (Image analysis)
- pandas (Data manipulation)
- matplotlib (Plotting)
- seaborn (Visualizations)
- joblib (Model persistence)
- tqdm (Progress bars)

**Note:** Installation may take 5-10 minutes depending on your system.

### Step 5: Verify Installation

Run the installation test script:

```bash
python test_installation.py
```

If successful, you'll see:
```
🎉 SUCCESS! Installation is complete and working correctly.
```

If there are errors, see the [Troubleshooting](#troubleshooting) section.

---

## Verification

### Test 1: Check Python Environment

```bash
python -c "import sklearn, numpy, cv2; print('All packages imported successfully!')"
```

### Test 2: Check Project Structure

```bash
ls src/
# Should show: feature_extractor.py, data_loader.py, train_svm.py, predict.py, organize_dataset.py
```

### Test 3: Run Quick Start

```bash
python quick_start.py
```

Follow the interactive prompts to create a synthetic dataset and train a test model.

---

## Troubleshooting

### Issue: "python: command not found"

**Solution:** Use `python3` instead of `python`:
```bash
python3 -m venv venv
python3 test_installation.py
```

### Issue: "pip: command not found"

**Solution:** Use `pip3` or install pip:
```bash
# Try pip3
pip3 install -r requirements.txt

# Or install pip
python3 -m ensurepip --upgrade
```

### Issue: "No module named 'cv2'" after installation

**Solution:** Reinstall opencv-python:
```bash
pip uninstall opencv-python
pip install opencv-python
```

### Issue: Permission denied errors

**Solution:** Use `--user` flag (not recommended with venv):
```bash
pip install --user -r requirements.txt
```

Or use sudo (not recommended):
```bash
sudo pip install -r requirements.txt
```

**Better solution:** Use a virtual environment (see Step 2).

### Issue: SSL Certificate errors

**Solution:** Try with trusted host:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Issue: "error: Microsoft Visual C++ 14.0 or greater is required" (Windows)

**Solution:** Install Microsoft C++ Build Tools:
1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"
3. Restart your terminal
4. Try installation again

### Issue: Memory errors during training

**Solution:**
- Reduce image size: Edit config in `train_svm.py`: `"image_size": (128, 128)`
- Use fewer images for testing
- Close other applications
- Disable some features temporarily

### Issue: Import errors with custom modules

**Solution:** Ensure you're in the project root directory:
```bash
cd svmTest
python test_installation.py
```

### Issue: "ModuleNotFoundError" for project modules

**Solution:** Add project to Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # On Unix/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%\src          # On Windows CMD
```

---

## Platform-Specific Notes

### macOS

**Install Command Line Tools (if needed):**
```bash
xcode-select --install
```

**Use Python 3 explicitly:**
```bash
python3 -m venv venv
python3 test_installation.py
```

**If using Homebrew Python:**
```bash
brew install python
python3 -m venv venv
```

### Linux

**Ubuntu/Debian:**
```bash
# Install Python development headers
sudo apt-get update
sudo apt-get install python3-dev python3-pip python3-venv

# Install system dependencies for OpenCV
sudo apt-get install libopencv-dev python3-opencv

# Create venv and install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Fedora/RHEL:**
```bash
sudo dnf install python3-devel python3-pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows

**Using Anaconda (Alternative Method):**
```bash
conda create -n svm_plants python=3.9
conda activate svm_plants
conda install scikit-learn numpy scipy pillow pandas matplotlib seaborn
pip install opencv-python scikit-image tqdm joblib
```

**PowerShell Execution Policy (if needed):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Alternative Installation Methods

### Method 1: Install Individual Packages

If the requirements.txt fails, install packages one by one:

```bash
pip install scikit-learn
pip install numpy
pip install scipy
pip install Pillow
pip install opencv-python
pip install scikit-image
pip install pandas
pip install matplotlib
pip install seaborn
pip install joblib
pip install tqdm
```

### Method 2: Use Conda

```bash
conda create -n svm_plants python=3.9
conda activate svm_plants
conda install -c conda-forge scikit-learn numpy scipy pillow opencv pandas matplotlib seaborn
pip install scikit-image tqdm joblib
```

### Method 3: Minimal Installation (Core Only)

For basic functionality without visualizations:

```bash
pip install scikit-learn numpy scipy Pillow opencv-python scikit-image joblib
```

---

## Upgrading

To upgrade to a newer version of dependencies:

```bash
pip install --upgrade -r requirements.txt
```

To upgrade a specific package:

```bash
pip install --upgrade scikit-learn
```

---

## Uninstallation

To remove the project and clean up:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv  # On Unix/Mac
rmdir /s venv  # On Windows

# Optionally remove the entire project
cd ..
rm -rf svmTest
```

---

## Docker Installation (Advanced)

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "test_installation.py"]
```

Build and run:
```bash
docker build -t svm-plants .
docker run -it svm-plants
```

---

## Next Steps

After successful installation:

1. ✅ **Read the Getting Started Guide:**
   ```bash
   cat GETTING_STARTED.md
   ```

2. ✅ **Run the Quick Start:**
   ```bash
   python quick_start.py
   ```

3. ✅ **Explore Examples:**
   ```bash
   python example_usage.py
   ```

4. ✅ **Read Full Documentation:**
   ```bash
   cat README.md
   ```

---

## Getting Help

If you encounter issues not covered here:

1. Check `README.md` for detailed documentation
2. Review `GETTING_STARTED.md` for usage help
3. Run `python test_installation.py` to diagnose issues
4. Check the [Troubleshooting](#troubleshooting) section above
5. Open an issue on GitHub with:
   - Your OS and Python version
   - Full error message
   - Output of `pip list`

---

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.9+ |
| RAM | 2GB | 4GB+ |
| Disk Space | 500MB | 1GB+ |
| CPU | Dual-core | Quad-core+ |
| OS | Windows 10, macOS 10.14, Linux | Latest versions |

---

**Installation complete! Ready to classify plants! 🌱**