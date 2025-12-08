# Getting Started with SVM Plant Classification

Welcome! This guide will help you get started with training an SVM model to classify plant images in just a few minutes.

## 🎯 What You'll Learn

By the end of this guide, you'll have:
- Set up the project environment
- Organized your plant images
- Trained your first SVM classifier
- Made predictions on new images

## ⏱️ Estimated Time: 10-15 minutes

---

## Step 1: Install Dependencies (2 minutes)

First, make sure you have Python 3.8 or higher installed:

```bash
python --version
```

Install all required packages:

```bash
cd svmTest
pip install -r requirements.txt
```

**Tip:** It's recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Step 2: Prepare Your Data (5 minutes)

You have four options:

### Option A: Quick Test with Synthetic Data ⚡ (Fastest)

Perfect for learning how the system works:

```bash
cd src
python train_svm.py --create-synthetic
```

This creates fake images for testing. Skip to **Step 3**.

### Option B: Use Before/After Directory Structure 🎯 (Recommended for Sorted Data)

If you have already sorted your target species from a larger collection:

1. Create the directory structure:
```bash
mkdir -p src/load/before
mkdir -p src/load/after
```

2. Place your images:
   - **`src/load/before/`** - All images (unsorted collection)
   - **`src/load/after/`** - Only target species images (manually sorted)

3. Run the organization script:
```bash
cd src
python organize_dataset.py --from-before-after --load-dir src/load
```

The script will automatically identify non-target images as those in 'before' but not in 'after' (compared by filename).

**This is perfect when you've already manually sorted out your target species!**

### Option C: Use the Interactive Organizer 🔧

If you have separate directories for each class:

```bash
cd src
python organize_dataset.py --interactive
```

Follow the prompts to organize your images. You'll need:
- Images of your **target species** (the plant you want to identify)
- Images of **other species** (plants that are NOT your target)

### Option D: Manual Organization 📁

Create this folder structure and add your images:

```
data/
├── train/
│   ├── target_species/     # 80% of your target species images
│   └── other_species/      # 80% of your other species images
└── test/
    ├── target_species/     # 20% of your target species images
    └── other_species/      # 20% of your other species images
```

**Minimum recommended:** 50 images per class (target + other)

---

## Step 3: Train Your Model (5 minutes)

Navigate to the `src` directory and start training:

```bash
cd src
python train_svm.py
```

**For faster training** (skips hyperparameter optimization):

```bash
python train_svm.py --no-grid-search
```

### What's Happening?

The script will:
1. Extract features from your images (colors, shapes, textures)
2. Train an SVM classifier
3. Evaluate performance on test images
4. Save the trained model

### Expected Output

You'll see something like:
```
Training Accuracy: 0.9500
Cross-Validation Accuracy: 0.9200 (+/- 0.0300)
Test Set Performance:
  Accuracy:  0.9100
  Precision: 0.9000
  Recall:    0.9200
  F1 Score:  0.9100
```

---

## Step 4: Make Predictions (2 minutes)

Use your trained model to classify new images:

### Single Image

```bash
python predict.py \
    --model ../models/svm_plant_classifier_TIMESTAMP.pkl \
    --images path/to/your/image.jpg
```

Replace `TIMESTAMP` with the actual timestamp of your saved model.

### Multiple Images

```bash
python predict.py \
    --model ../models/svm_plant_classifier_TIMESTAMP.pkl \
    --directory path/to/image/folder \
    --recursive
```

### With Visualizations

```bash
python predict.py \
    --model ../models/svm_plant_classifier_TIMESTAMP.pkl \
    --images image.jpg \
    --visualize \
    --viz-dir ../predictions
```

---

## 🎓 Quick Start Script

Want everything automated? Use our quick start script:

```bash
python quick_start.py
```

This interactive script will guide you through the entire process!

---

## 📊 Understanding Your Results

After training, check these files:

1. **`results/confusion_matrix.png`** - Shows correct vs incorrect predictions
2. **`results/roc_curve.png`** - Shows model's discrimination ability
3. **`results/results_TIMESTAMP.json`** - Detailed metrics

### What Do the Metrics Mean?

- **Accuracy**: Overall correctness (higher is better)
- **Precision**: Of predicted target species, how many are actually correct?
- **Recall**: Of actual target species, how many did we find?
- **F1-Score**: Balance between precision and recall

**Good performance:** All metrics above 0.80 (80%)

---

## 🚀 Next Steps

Now that you have a working model, try:

1. **Add More Training Data**
   - More images = better performance
   - Aim for 100+ images per class

2. **Balance Your Classes**
   - Try to have similar numbers of target and other species images
   - Imbalanced data can hurt performance

3. **Improve Image Quality**
   - Use clear, well-lit photos
   - Include various angles and backgrounds
   - Remove blurry or unclear images

4. **Experiment with Features**
   - Edit the config in `train_svm.py`
   - Try different combinations of features
   - Run `example_usage.py` Example 6

5. **Fine-tune Hyperparameters**
   - Run training WITH grid search (remove `--no-grid-search`)
   - Adjust the parameter grid in `train_svm.py`

---

## 💡 Tips for Success

### For Best Results:

✅ **Use diverse images**: Different lighting, angles, backgrounds
✅ **Balance your dataset**: Similar number of images per class
✅ **Clean your data**: Remove duplicates and poor-quality images
✅ **Start simple**: Test with synthetic data first
✅ **Iterate**: Add more data based on what the model gets wrong

### Common Issues:

❌ **Low accuracy (<60%)**
   - Add more training images
   - Check if images are labeled correctly
   - Ensure good image quality

❌ **High training accuracy but low test accuracy**
   - Model is overfitting
   - Add more diverse training data
   - Try simpler features

❌ **Model predicts mostly one class**
   - Classes are imbalanced
   - Add more images of the minority class
   - Or use class weighting

---

## 🆘 Need Help?

### Check These First:

1. **Project Structure**: Make sure directories exist
2. **Image Format**: Use JPG, PNG, or BMP
3. **Python Version**: Must be 3.8 or higher
4. **Dependencies**: Run `pip install -r requirements.txt` again

### Still Stuck?

- Check the full **README.md** for detailed documentation
- Review **example_usage.py** for code examples
- Look at the **Troubleshooting** section in README.md

---

## 📖 Example Workflow

Here's a complete example from start to finish:

```bash
# 1. Setup
cd svmTest
pip install -r requirements.txt

# 2. Organize data using before/after structure
mkdir -p src/load/before src/load/after
# (Place all images in 'before', target species in 'after')
cd src
python organize_dataset.py --from-before-after --load-dir src/load

# 3. Train model (fast mode)
python train_svm.py --no-grid-search

# 4. Predict on new images
python predict.py \
    --model ../models/svm_plant_classifier_20240101_120000.pkl \
    --directory ../data/test/target_species \
    --visualize

# 5. View results
open ../results/confusion_matrix.png
open ../predictions/
```

---

## 🎉 Congratulations!

You've successfully:
- ✅ Set up the SVM plant classifier
- ✅ Trained your first model
- ✅ Made predictions on plant images

You're now ready to classify plant species with machine learning!

### What's Next?

- Experiment with different features and parameters
- Add more training data to improve accuracy
- Try the advanced examples in `example_usage.py`
- Read the full documentation in `README.md`

---

**Happy Classifying! 🌱🔬**