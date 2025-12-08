# Before/After Organization Guide

## Overview

The **before/after** organization feature allows you to automatically separate target and non-target images when you've already manually sorted your target species from a larger collection. This is perfect for workflows where you:

1. Start with a large collection of mixed images
2. Manually identify and separate your target species
3. Want to automatically create training/test splits for both classes

> **📍 Important: Working Directory**
>
> This guide assumes you're working from the **project root** (`svmTest/`) unless otherwise specified. All commands starting with `python src/...` should be run from the project root. If you prefer to work from the `src/` directory, adjust paths accordingly (e.g., `python organize_dataset.py` instead of `python src/organize_dataset.py`).
</text>


## 🎯 Use Case

**Traditional Workflow Problem:**
- You have 1000 insect photos
- You manually sort through and identify 200 monarch butterflies
- Now you need to organize them for machine learning
- Manually creating train/test splits is tedious and error-prone

**Before/After Solution:**
- Place all 1000 photos in `before/` directory
- Copy the 200 monarch butterflies to `after/` directory
- Run one command
- Automatically get organized train/test splits for both classes!

## 📁 Directory Structure

```
svmTest/
└── src/
    └── load/
        ├── before/         # All images (unsorted collection)
        │   ├── image_001.jpg
        │   ├── image_002.jpg
        │   ├── ...
        │   └── image_1000.jpg
        └── after/          # Only target species (manually sorted)
            ├── image_005.jpg    # Subset of 'before'
            ├── image_017.jpg
            ├── ...
            └── image_892.jpg
```

## 🚀 Quick Start

### Step 1: Create Directories

```bash
cd svmTest
mkdir -p src/load/before
mkdir -p src/load/after
```

### Step 2: Add Your Images

1. **Place ALL images in `before/`**
   ```bash
   cp /path/to/all/images/* src/load/before/
   ```

2. **Copy ONLY target species to `after/`**
   ```bash
   # Manually review and copy target species
   cp src/load/before/target_001.jpg src/load/after/
   cp src/load/before/target_015.jpg src/load/after/
   # ... etc
   ```

### Step 3: Run Organization Script

**Option A: From project root** (recommended for beginners):

```bash
cd svmTest
python src/organize_dataset.py --from-before-after --load-dir src/load
```

**Option B: From src directory**:

```bash
cd svmTest/src
python organize_dataset.py --from-before-after --load-dir load
```

**That's it!** Your images are now organized into:
- `data/train/target_species/` - Target species training set
- `data/train/other_species/` - Other species training set
- `data/test/target_species/` - Target species test set
- `data/test/other_species/` - Other species test set

## 🔍 How It Works

### Automatic Classification

The script:
1. Reads all images from `before/` directory
2. Reads all images from `after/` directory
3. **Compares filenames** to identify:
   - **Target species**: Images present in `after/`
   - **Other species**: Images in `before/` but NOT in `after/`
4. Splits both classes into training (80%) and test (20%) sets
5. Copies/moves files to proper `data/` structure

### Example

```
Input:
  before/ has:  [img_1.jpg, img_2.jpg, img_3.jpg, img_4.jpg, img_5.jpg]
  after/ has:   [img_2.jpg, img_4.jpg]

Processing:
  Target species:  img_2.jpg, img_4.jpg          (2 images)
  Other species:   img_1.jpg, img_3.jpg, img_5.jpg  (3 images)

Output:
  data/train/target_species/  : ~1-2 images (80% of 2)
  data/test/target_species/   : ~0-1 images (20% of 2)
  data/train/other_species/   : ~2 images (80% of 3)
  data/test/other_species/    : ~1 image (20% of 3)
```

## 📝 Complete Example

### Scenario: Bird Classification

You want to identify Cedar Waxwings from your bird photos.

```bash
# 1. Setup
cd svmTest
mkdir -p src/load/before src/load/after

# 2. Copy all bird photos to 'before'
cp ~/Photos/Birds/*.jpg src/load/before/
# Result: 500 bird photos in 'before/'

# 3. Review and identify Cedar Waxwings
# Use your favorite image viewer to go through photos
# Copy Cedar Waxwings to 'after/'
cp src/load/before/IMG_2301.jpg src/load/after/
cp src/load/before/IMG_2315.jpg src/load/after/
cp src/load/before/IMG_2387.jpg src/load/after/
# ... continue for all Cedar Waxwings
# Result: 75 Cedar Waxwing photos in 'after/'

# 4. Organize automatically (from project root)
python src/organize_dataset.py --from-before-after --load-dir src/load

# 5. Verify organization
python src/organize_dataset.py --stats

# 6. Train your model
python src/train_svm.py --data-root data
```

**Output:**
```
Training Set:
  Target Species:     60 images  (Cedar Waxwings, 80% of 75)
  Other Species:     340 images  (Other birds, 80% of 425)
  Total:             400 images

Test Set:
  Target Species:     15 images  (Cedar Waxwings, 20% of 75)
  Other Species:      85 images  (Other birds, 20% of 425)
  Total:             100 images
```

## ⚙️ Command Options

### Basic Usage

From project root:
```bash
cd svmTest
python src/organize_dataset.py --from-before-after --load-dir src/load
```

Or from src directory:
```bash
cd svmTest/src
python organize_dataset.py --from-before-after --load-dir load
```

### Custom Train/Test Split

```bash
# 70% training, 30% test (from src directory)
cd svmTest/src
python organize_dataset.py \
    --from-before-after \
    --load-dir load \
    --test-size 0.3
```

### Move Instead of Copy

**⚠️ Warning: This removes files from source directories!**

```bash
cd svmTest/src
python organize_dataset.py \
    --from-before-after \
    --load-dir load \
    --move
```

### Custom Output Directory

```bash
cd svmTest/src
python organize_dataset.py \
    --from-before-after \
    --load-dir load \
    --data-root /path/to/custom/output
```

### Different Load Directory

```bash
cd svmTest/src
python organize_dataset.py \
    --from-before-after \
    --load-dir /path/to/my/before_after_structure
```

## ✅ Best Practices

### DO:

✅ **Use exact same filenames** in both directories
- `before/butterfly_01.jpg` → `after/butterfly_01.jpg` ✓

✅ **Keep 'before' as backup** - Copy files, don't move them

✅ **Test with small sample first** - Try 10-20 images to verify

✅ **Check output with `--stats`** - Verify organization is correct

✅ **Use consistent image formats** - All JPG or all PNG

### DON'T:

❌ **Rename files** between directories
- `before/img_01.jpg` → `after/butterfly_01.jpg` ✗

❌ **Have files only in 'after'** - All 'after' files must exist in 'before'

❌ **Use `--move` without backup** - Always keep your original images safe

❌ **Mix different projects** - One before/after structure per classification task

## 🐛 Troubleshooting

### Error: "No non-target images found"

**Problem:** All images in 'before' are also in 'after'

**Solution:** 
- Ensure 'after' contains ONLY target species
- Not ALL images from 'before'

```bash
# Wrong:
before/ has: [1.jpg, 2.jpg, 3.jpg]
after/ has:  [1.jpg, 2.jpg, 3.jpg]  # All images!

# Correct:
before/ has: [1.jpg, 2.jpg, 3.jpg]
after/ has:  [2.jpg]  # Only target species
```

### Error: "'before' directory not found"

**Problem:** Directory doesn't exist or wrong path

**Solution:**
```bash
# Check directories exist
ls -la src/load/

# Create if missing
mkdir -p src/load/before src/load/after

# Or specify correct path
python organize_dataset.py \
    --from-before-after \
    --load-dir /correct/path/to/load
```

### Error: "No images found in 'before' directory"

**Problem:** No images or unsupported format

**Solution:**
- Check supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`
- Verify images are directly in the directory (not in subdirectories)

```bash
# Check what's in there
ls src/load/before/

# Check file extensions
file src/load/before/*
```

### Issue: Filenames don't match

**Problem:** Images renamed between directories

**Example:**
```bash
before/ has: IMG_0001.jpg
after/ has:  monarch_butterfly.jpg  # Renamed!
```

**Solution:** Use exact same filenames
```bash
# Copy with original name
cp src/load/before/IMG_0001.jpg src/load/after/IMG_0001.jpg
```

### Issue: Train/test split seems wrong

**Problem:** Very small dataset

**Explanation:** 
- With < 10 images per class, splits may seem uneven
- This is normal behavior of the random split

**Solution:**
```bash
# Use fixed random seed for reproducibility
python organize_dataset.py \
    --from-before-after \
    --load-dir src/load \
    --random-state 42
```

## 💡 Tips & Tricks

### 1. Batch Copying Target Species

Use wildcards or file lists:

```bash
# If your target files have a pattern
cp src/load/before/monarch_*.jpg src/load/after/

# Or use a file list
while read filename; do
    cp "src/load/before/$filename" src/load/after/
done < target_species_list.txt
```

### 2. Verify Before Running

Check your setup:

```bash
echo "Before images: $(ls src/load/before/ | wc -l)"
echo "After images: $(ls src/load/after/ | wc -l)"
echo "Expected non-target: $(( $(ls src/load/before/ | wc -l) - $(ls src/load/after/ | wc -l) ))"
```

### 3. Dry Run (Test First)

Test with a small subset:

```bash
# Copy just 10 images to test
mkdir -p test/load/before test/load/after
cp src/load/before/img_{001..010}.jpg test/load/before/
cp src/load/before/img_{002,005,008}.jpg test/load/after/

# Test organization
python organize_dataset.py \
    --from-before-after \
    --load-dir test/load \
    --data-root test/data
```

### 4. Incremental Updates

Already organized but have more images?

```bash
# Navigate to project root
cd svmTest

# Clear previous organization (optional)
rm -rf data/train/* data/test/*

# Add new images to before/after
cp /new/images/*.jpg src/load/before/
# Sort new target species into after/

# Re-run organization
python src/organize_dataset.py --from-before-after --load-dir src/load
```

### 5. Keep Track of Your Sorting

Document which images you've sorted:

```bash
# Create a manifest (from project root)
cd svmTest
ls src/load/after/ > sorted_target_species.txt

# Later, check what you sorted
cat sorted_target_species.txt
```

## 🔄 Integration with Workflow

### Full ML Pipeline

**Important: Run all commands from the `svmTest/src/` directory** to ensure consistent paths.

```bash
# Navigate to src directory
cd svmTest/src

# 1. Organize data
python organize_dataset.py --from-before-after --load-dir load

# 2. Verify organization
python organize_dataset.py --stats

# 3. Train model
python train_svm.py --data-root ../data

# 4. Evaluate
python predict.py \
    --model ../models/svm_*.pkl \
    --directory ../data/test/target_species

# 5. If accuracy is low, add more training data
#    Add images to load/before and load/after, then repeat!
```

**Alternative: Run from project root** (if you prefer):

```bash
# Navigate to project root
cd svmTest

# 1. Organize data
python src/organize_dataset.py --from-before-after --load-dir src/load

# 2. Verify organization
python src/organize_dataset.py --stats

# 3. Train model
python src/train_svm.py --data-root data

# 4. Evaluate
python src/predict.py \
    --model models/svm_*.pkl \
    --directory data/test/target_species
```

## 📊 When to Use Before/After vs Other Methods

### Use Before/After When:
- ✅ You have a mixed collection and manually sorted target species
- ✅ Target species identification requires human expertise
- ✅ You want automated train/test splitting
- ✅ You're working with a curated subset of a larger dataset

### Use Interactive Mode When:
- ✅ You have separate directories already organized by class
- ✅ Multiple people contributed different classes
- ✅ Classes are already clearly separated

### Use Manual Organization When:
- ✅ You want complete control over train/test splits
- ✅ You have specific images that must be in test set
- ✅ Very small dataset where automation isn't needed

## 🎓 Learning Resources

- **Main README**: `README.md` - Complete project documentation
- **Getting Started**: `GETTING_STARTED.md` - Quick start tutorial
- **Load Directory README**: `src/load/README.md` - Detailed load structure guide
- **Script Help**: `python organize_dataset.py --help` - All command options

## 📧 Support

Having issues? Check:
1. Your directory structure matches the expected format
2. Filenames are identical in both directories
3. Images are in supported formats
4. Run `--stats` to verify output

---

**Happy Organizing! 📁✨**
