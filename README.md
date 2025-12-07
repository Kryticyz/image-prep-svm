# Binary Classification of pre sorted Images with Support Vector Machine (SVM)

This project was created with the support of Claude Sonnet 4.5 Thinking.
It was modified and adjusted to meet the authors needs.

Early versions of this codebase are clanker heavy and retain references to plant images. This project is intended to be flexible to any Image based Binary classifcation task where you have an image set previously processed by a human.

A supervised machine learning project for binary classification of pre sorted images using Support Vector Machines (SVM). This project identifies whether an image belongs to a target object (e.g., a plant species like *Myosotis sylvatica*) or is some other object.

## 🌟 Features

- **Multiple Feature Extraction Methods**:
  - Color histograms (RGB channels)
  - HOG (Histogram of Oriented Gradients)
  - LBP (Local Binary Pattern)
  - Statistical features (mean, std, skewness, kurtosis across RGB, HSV, LAB color spaces)

- **Automated Hyperparameter Tuning**: Grid search for optimal SVM parameters
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC
- **Visualization Tools**: Confusion matrices, ROC curves, precision-recall curves
- **Batch Prediction**: Classify single images or entire directories
- **Model Persistence**: Save and load trained models for reuse

## 📁 Project Structure

```
svmTest/
├── data/
│   ├── train/
│   │   ├── target_species/    # Training images of target species
│   │   └── other_species/     # Training images of other species
│   └── test/
│       ├── target_species/    # Test images of target species
│       └── other_species/     # Test images of other species
├── src/
│   ├── feature_extractor.py   # Feature extraction module
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── train_svm.py            # Training script
│   └── predict.py              # Prediction script
├── models/                     # Saved trained models
├── results/                    # Training results and visualizations
└── requirements.txt            # Python dependencies
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or navigate to the project directory:
```bash
cd svmTest
```

2. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### Option 1: Use Your Own Images

1. **Organize your images** into the following structure:
   ```
   data/
   ├── train/
   │   ├── target_species/    # Put target species training images here
   │   └── other_species/     # Put other species training images here
   └── test/
       ├── target_species/    # Put target species test images here
       └── other_species/     # Put other species test images here
   ```

2. **Supported formats**: JPG, JPEG, PNG, BMP, TIFF

3. **Recommended dataset size**:
   - Minimum: 50 images per class
   - Recommended: 200+ images per class for better performance
   - Test set: 20-30% of total data

### Option 2: Use Existing Data

If you have images in a different location (e.g., `src/Myosotis_sylvatica/`), you can organize them using Python:

```python
from src.data_loader import PlantImageDataLoader

loader = PlantImageDataLoader(data_root="data")
loader.organize_data_from_source(
    source_dir="src/Myosotis_sylvatica",
    test_size=0.2,
    class_label="target_species"
)
```

### Option 3: Generate Synthetic Data (for Testing)

To quickly test the pipeline:

```bash
cd src
python train_svm.py --create-synthetic
```

## 🎯 Usage

### Training a Model

Navigate to the `src` directory and run the training script:

```bash
cd src
python train_svm.py
```

**With custom parameters:**

```bash
python train_svm.py \
    --data-root ../data \
    --model-dir ../models \
    --results-dir ../results \
    --image-size 224 224
```

**Skip grid search (faster training):**

```bash
python train_svm.py --no-grid-search
```

**Training output includes:**
- Cross-validation scores
- Training and test accuracy
- Confusion matrix
- ROC curve
- Precision-recall curve
- Saved model file (.pkl)
- Results JSON file

### Making Predictions

**Classify a single image:**

```bash
cd src
python predict.py \
    --model ../models/svm_plant_classifier_20240101_120000.pkl \
    --images path/to/image.jpg
```

**Classify multiple images:**

```bash
python predict.py \
    --model ../models/svm_plant_classifier_20240101_120000.pkl \
    --images image1.jpg image2.jpg image3.jpg
```

**Classify all images in a directory:**

```bash
python predict.py \
    --model ../models/svm_plant_classifier_20240101_120000.pkl \
    --directory path/to/images \
    --recursive
```

**Generate visualizations:**

```bash
python predict.py \
    --model ../models/svm_plant_classifier_20240101_120000.pkl \
    --directory path/to/images \
    --visualize \
    --viz-dir ../predictions
```

**Save predictions to JSON:**

```bash
python predict.py \
    --model ../models/svm_plant_classifier_20240101_120000.pkl \
    --directory path/to/images \
    --output predictions.json
```

## ⚙️ Configuration

### Feature Extraction Parameters

You can customize feature extraction in `train_svm.py`:

```python
config = {
    "image_size": (224, 224),           # Image resize dimensions
    "use_color_histogram": True,        # RGB color histograms
    "use_hog": True,                    # HOG features
    "use_lbp": True,                    # Local Binary Patterns
    "use_statistics": True,             # Statistical features
    "svm_kernel": "rbf",                # SVM kernel: 'rbf', 'linear', 'poly'
    "svm_C": 1.0,                       # Regularization parameter
    "svm_gamma": "scale",               # Kernel coefficient
    "perform_grid_search": True,        # Hyperparameter tuning
    "cross_validation_folds": 5,        # CV folds
    "random_state": 42,                 # Random seed
}
```

### Grid Search Parameters

The grid search explores the following parameter space:

- **C**: [0.1, 1, 10, 100]
- **gamma**: ['scale', 'auto', 0.001, 0.01, 0.1]
- **kernel**: ['rbf', 'linear']

## 📈 Understanding the Results

### Training Metrics

- **Training Accuracy**: Performance on training set
- **Cross-Validation Accuracy**: Average accuracy across CV folds
- **Best Parameters**: Optimal hyperparameters found by grid search

### Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: Of predicted target species, how many are correct?
- **Recall**: Of actual target species, how many were identified?
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (discrimination ability)

### Visualizations

1. **Confusion Matrix**: Shows true vs predicted labels
2. **ROC Curve**: True positive rate vs false positive rate
3. **Precision-Recall Curve**: Trade-off between precision and recall
4. **Prediction Distribution**: Histogram of predicted probabilities

## 🔬 Technical Details

### Feature Extraction

The system extracts a comprehensive feature vector from each image:

1. **Color Histogram Features** (96 dimensions):
   - 32-bin histograms for R, G, B channels
   - Captures color distribution

2. **HOG Features** (~5000+ dimensions):
   - Captures shape and edge information
   - 9 orientations, 8×8 pixels per cell

3. **LBP Features** (26 dimensions):
   - Texture descriptor
   - Robust to illumination changes

4. **Statistical Features** (54 dimensions):
   - Mean, std, min, max, skewness, kurtosis
   - Computed for RGB, HSV, and LAB color spaces

**Total feature dimension**: ~5,200 features per image

### SVM Classification

- **Algorithm**: Support Vector Machine with RBF kernel
- **Advantages**:
  - Effective in high-dimensional spaces
  - Memory efficient (uses support vectors)
  - Versatile through different kernel functions
  - Good performance with limited training data

### Preprocessing

1. Image resizing to 224×224 pixels
2. Feature normalization using StandardScaler
3. Maintains aspect ratio during preprocessing

## 🎓 Example Workflow

Here's a complete workflow from start to finish:

```bash
# 1. Navigate to src directory
cd svmTest/src

# 2. Check dataset statistics
python -c "from data_loader import PlantImageDataLoader; \
           loader = PlantImageDataLoader('../data'); \
           loader.print_dataset_summary()"

# 3. Train the model
python train_svm.py --data-root ../data

# 4. Predict on new images
python predict.py \
    --model ../models/svm_plant_classifier_YYYYMMDD_HHMMSS.pkl \
    --directory ../data/test/target_species \
    --visualize \
    --output ../results/predictions.json

# 5. View results in the results/ and predictions/ directories
```

## 📋 Requirements

### Core Libraries

- **scikit-learn**: SVM implementation and ML utilities
- **numpy**: Numerical computations
- **scipy**: Scientific computing
- **Pillow**: Image loading and processing
- **opencv-python**: Computer vision operations
- **scikit-image**: Advanced image processing
- **pandas**: Data manipulation
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical visualizations
- **joblib**: Model serialization
- **tqdm**: Progress bars

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'cv2'`
- **Solution**: Install OpenCV: `pip install opencv-python`

**Issue**: `FileNotFoundError: No images found`
- **Solution**: Ensure images are in the correct directories and have supported extensions

**Issue**: Low accuracy on test set
- **Solution**: 
  - Increase training dataset size
  - Check for class imbalance
  - Try different feature combinations
  - Adjust hyperparameters

**Issue**: Memory errors during training
- **Solution**:
  - Reduce image size: `--image-size 128 128`
  - Disable some features (edit config in `train_svm.py`)
  - Process fewer images at once

**Issue**: Training takes too long
- **Solution**:
  - Use `--no-grid-search` flag
  - Reduce image size
  - Reduce dataset size for quick testing

## 🔮 Future Improvements

- [ ] Add data augmentation (rotation, flip, brightness)
- [ ] Implement ensemble methods (combine multiple models)
- [ ] Add support for multi-class classification (>2 species)
- [ ] Integrate deep learning features (CNN embeddings)
- [ ] Create web interface for easy predictions
- [ ] Add LIME/SHAP for model interpretability
- [ ] Implement active learning for labeling
- [ ] Support for video classification

## 📝 Citation

If you use this project in your research, please cite:

```
Binary Classification with SVM
Binary classification of images using Support Vector Machines
GitHub: https://github.com/Kryticyz/image-prep-svm
```

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy Classifying! 🌱**
