# System Architecture

This document describes the architecture of the SVM Plant Classification system.

## 📐 Overview

The system follows a modular, pipeline-based architecture for image classification using Support Vector Machines (SVM). The design separates concerns into distinct layers: data management, feature extraction, model training, and prediction.

---

## 🏗️ System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  (CLI Scripts: train_svm.py, predict.py, organize_dataset.py)  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ SVMPlant        │  │ PlantImage       │  │ ImageFeature  │ │
│  │ Classifier      │  │ DataLoader       │  │ Extractor     │ │
│  └─────────────────┘  └──────────────────┘  └───────────────┘ │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PROCESSING LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐│
│  │ Feature    │  │ Image    │  │ Model    │  │ Evaluation   ││
│  │ Engineering│  │ Preprocessing│ Training │  │ & Metrics   ││
│  └────────────┘  └──────────┘  └──────────┘  └──────────────┘│
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ML LIBRARY LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  scikit-learn │ OpenCV │ scikit-image │ NumPy │ SciPy         │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  Images │ Models (.pkl) │ Results (JSON/PNG) │ Datasets        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Core Components

### 1. Feature Extractor (`feature_extractor.py`)

**Purpose:** Transforms raw images into numerical feature vectors.

**Key Classes:**
- `ImageFeatureExtractor`: Main feature extraction engine

**Responsibilities:**
- Load and preprocess images
- Extract color histogram features (RGB)
- Extract HOG (Histogram of Oriented Gradients) features
- Extract LBP (Local Binary Pattern) features
- Extract statistical features (RGB, HSV, LAB color spaces)
- Combine features into unified vector

**Inputs:** Image file path
**Outputs:** Feature vector (numpy array, ~5200 dimensions)

**Dependencies:**
- OpenCV (cv2): Image processing
- scikit-image: HOG and LBP extraction
- Pillow (PIL): Image loading
- NumPy/SciPy: Numerical operations

---

### 2. Data Loader (`data_loader.py`)

**Purpose:** Manages dataset organization and loading.

**Key Classes:**
- `PlantImageDataLoader`: Dataset management

**Responsibilities:**
- Organize images into train/test splits
- Load images from directories
- Generate dataset statistics
- Handle synthetic data generation
- Batch feature extraction

**Inputs:** Directory paths, image files
**Outputs:** Feature arrays (X), labels (y), file paths

**Dependencies:**
- ImageFeatureExtractor: Feature extraction
- scikit-learn: Train/test splitting
- NumPy: Array operations

---

### 3. SVM Classifier (`train_svm.py`)

**Purpose:** Train and evaluate SVM models.

**Key Classes:**
- `SVMPlantClassifier`: Main classifier

**Responsibilities:**
- Configure feature extraction pipeline
- Train SVM with hyperparameter tuning
- Perform cross-validation
- Evaluate model performance
- Save/load models
- Generate predictions

**Inputs:** Training data (X, y)
**Outputs:** Trained model, metrics, predictions

**Dependencies:**
- scikit-learn: SVM, GridSearchCV, metrics
- Feature extractor: Feature generation
- joblib: Model persistence

---

### 4. Prediction Engine (`predict.py`)

**Purpose:** Use trained models for inference.

**Key Classes:**
- `PlantImagePredictor`: Prediction interface

**Responsibilities:**
- Load trained models
- Predict single images
- Batch prediction
- Generate visualizations
- Export results

**Inputs:** Trained model, image files
**Outputs:** Predictions, probabilities, visualizations

**Dependencies:**
- SVMPlantClassifier: Model loading
- matplotlib: Visualizations
- JSON: Results export

---

### 5. Dataset Organizer (`organize_dataset.py`)

**Purpose:** Prepare and organize image datasets.

**Responsibilities:**
- Interactive dataset organization
- Train/test splitting
- File management (copy/move)
- Dataset statistics
- Validation

**Inputs:** Source directories, configuration
**Outputs:** Organized dataset structure

---

## 📊 Data Flow

### Training Pipeline

```
1. Raw Images
   └─> data/train/target_species/*.jpg
   └─> data/train/other_species/*.jpg

2. Data Loading (data_loader.py)
   └─> Load images from directories
   └─> Apply train/test split

3. Feature Extraction (feature_extractor.py)
   ├─> Resize images (224x224)
   ├─> Extract color histograms
   ├─> Extract HOG features
   ├─> Extract LBP features
   └─> Extract statistical features
   └─> Concatenate → Feature Vector (~5200D)

4. Preprocessing
   └─> StandardScaler: Normalize features
   └─> X_scaled, y_labels

5. Training (train_svm.py)
   ├─> Grid Search (optional)
   │   └─> Try different C, gamma, kernel
   ├─> Cross-validation
   └─> Train best SVM model

6. Evaluation
   ├─> Predict on test set
   ├─> Calculate metrics
   └─> Generate visualizations

7. Model Persistence
   └─> Save to models/*.pkl
   └─> Save results to results/*.json
```

### Prediction Pipeline

```
1. New Image
   └─> path/to/image.jpg

2. Load Trained Model
   └─> models/svm_classifier.pkl
   └─> Includes: model, scaler, feature_extractor

3. Feature Extraction
   └─> Same pipeline as training
   └─> Extract all features

4. Preprocessing
   └─> Apply saved scaler

5. Prediction
   ├─> SVM.predict() → class label
   └─> SVM.predict_proba() → probabilities

6. Output
   └─> Class: Target Species / Other Species
   └─> Confidence: 0.XX
   └─> Visualization (optional)
```

---

## 🎯 Design Patterns

### 1. Strategy Pattern
- Feature extraction strategies can be enabled/disabled
- Different SVM kernels can be selected
- Configurable through settings

### 2. Pipeline Pattern
- Sequential processing stages
- Each stage transforms data for next stage
- Clear data flow from images → features → predictions

### 3. Factory Pattern
- `ImageFeatureExtractor` creates different feature extractors
- Configuration-based instantiation

### 4. Singleton-like Behavior
- Feature extractor configuration is set once
- Reused across all images for consistency

---

## 🔄 Module Dependencies

```
┌─────────────────────┐
│   train_svm.py      │
│   predict.py        │
│   organize_dataset  │
└──────────┬──────────┘
           │
           ├──────────────────────────┐
           │                          │
           ▼                          ▼
┌──────────────────────┐    ┌────────────────────┐
│   data_loader.py     │    │  train_svm.py      │
│   (PlantImage        │    │  (SVMPlant         │
│    DataLoader)       │    │   Classifier)      │
└──────────┬───────────┘    └────────┬───────────┘
           │                         │
           └────────┬────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ feature_extractor   │
         │ (ImageFeature       │
         │  Extractor)         │
         └──────────┬──────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐      ┌──────────────────┐
│  scikit-learn │      │  OpenCV          │
│  scikit-image │      │  PIL             │
│  numpy, scipy │      │  matplotlib      │
└───────────────┘      └──────────────────┘
```

---

## 💾 Data Structures

### Feature Vector Structure

```python
feature_vector = [
    # Color Histogram (96 features)
    R_hist[0...31],    # Red channel histogram (32 bins)
    G_hist[0...31],    # Green channel histogram (32 bins)
    B_hist[0...31],    # Blue channel histogram (32 bins)
    
    # HOG Features (~5000+ features)
    hog_features[0...N],
    
    # LBP Features (26 features)
    lbp_histogram[0...25],
    
    # Statistical Features (54 features)
    # For each color space (RGB, HSV, LAB):
    #   For each channel (3):
    #     mean, std, min, max, skewness, kurtosis (6)
    # Total: 3 spaces × 3 channels × 6 stats = 54
    stats[0...53]
]
```

### Model Object Structure

```python
model_object = {
    'model': SVC_instance,           # Trained SVM
    'scaler': StandardScaler,        # Feature normalizer
    'feature_extractor': ImageFeatureExtractor,  # Feature pipeline
    'config': dict,                  # Training configuration
    'training_history': dict         # Training metrics
}
```

### Prediction Result Structure

```python
prediction_result = {
    'image_path': str,
    'prediction': int,               # 0 or 1
    'class_name': str,               # "Target Species" or "Other Species"
    'probability': {
        'other_species': float,      # 0.0 to 1.0
        'target_species': float      # 0.0 to 1.0
    },
    'confidence': float              # max(probabilities)
}
```

---

## 🔐 Configuration Management

Configuration is managed through Python dictionaries:

```python
config = {
    # Image settings
    'image_size': (224, 224),
    
    # Feature extraction toggles
    'use_color_histogram': bool,
    'use_hog': bool,
    'use_lbp': bool,
    'use_statistics': bool,
    
    # SVM parameters
    'svm_kernel': str,              # 'rbf', 'linear', 'poly'
    'svm_C': float,                 # Regularization
    'svm_gamma': str/float,         # Kernel coefficient
    
    # Training settings
    'perform_grid_search': bool,
    'cross_validation_folds': int,
    'random_state': int
}
```

---

## 🎛️ Extension Points

The architecture is designed to be extensible:

### 1. Adding New Features
```python
# In feature_extractor.py
def _extract_new_feature(self, image):
    # Implement new feature
    return features

# In extract_features()
if self.use_new_feature:
    new_features = self._extract_new_feature(image)
    features.append(new_features)
```

### 2. Adding New Classifiers
```python
# Create new_classifier.py
class NewClassifier:
    def train(self, X, y): ...
    def predict(self, X): ...
```

### 3. Adding New Data Sources
```python
# Extend data_loader.py
class DataLoader:
    def load_from_database(self): ...
    def load_from_api(self): ...
```

---

## 📈 Scalability Considerations

### Current Implementation
- **Scale:** Up to ~10,000 images
- **Memory:** Loads all features in memory
- **Processing:** Single-threaded feature extraction

### Future Improvements
- **Batch Processing:** Process images in batches
- **Parallel Extraction:** Use multiprocessing
- **Lazy Loading:** Load features on-demand
- **Database:** Store features in database
- **Distributed:** Use Dask/Ray for large datasets

---

## 🔒 Error Handling

### Levels of Error Handling

1. **Input Validation**
   - Check file existence
   - Validate image formats
   - Verify directory structure

2. **Processing Errors**
   - Try/catch in feature extraction
   - Skip corrupted images
   - Log errors for review

3. **Model Errors**
   - Validate training data
   - Check for convergence
   - Handle prediction failures

4. **Output Errors**
   - Ensure directories exist
   - Handle file write failures
   - Validate saved models

---

## 🧪 Testing Strategy

### Unit Tests (Potential)
- Feature extraction for single image
- Data loading functions
- Model save/load functionality

### Integration Tests
- End-to-end training pipeline
- Prediction pipeline
- Data organization workflow

### System Tests
- `test_installation.py`: Verify setup
- `example_usage.py`: Test examples
- `quick_start.py`: Full workflow test

---

## 📦 Deployment Architecture

### Local Deployment (Current)
```
User Machine
└─> Python Environment
    └─> CLI Scripts
        └─> Local Files
```

### Potential Web Deployment
```
Client (Browser)
    ↓
Web Server (Flask/FastAPI)
    ↓
SVM Prediction Service
    ↓
Model Storage
```

### Potential API Deployment
```
Client Application
    ↓ HTTP/REST
API Gateway
    ↓
Prediction Microservice
    ├─> Model Cache
    └─> Result Store
```

---

## 🔍 Performance Characteristics

### Time Complexity
- **Feature Extraction:** O(n × m) where n = images, m = pixels
- **Training:** O(n² × d) where n = samples, d = features (SVM)
- **Prediction:** O(k × d) where k = support vectors

### Space Complexity
- **Features:** O(n × d) where n = images, d = ~5200
- **Model:** O(k × d) where k = support vectors
- **Cache:** Minimal (loads images one at a time)

---

## 🎯 Design Principles

1. **Modularity:** Each component has single responsibility
2. **Configurability:** Easy to adjust parameters
3. **Extensibility:** Simple to add new features
4. **Reusability:** Components can be used independently
5. **Maintainability:** Clear code structure and documentation
6. **Testability:** Components can be tested in isolation

---

## 📚 Technology Stack Summary

| Layer | Technologies |
|-------|-------------|
| **Interface** | Python CLI (argparse) |
| **Application** | Custom Python classes |
| **ML/CV** | scikit-learn, OpenCV, scikit-image |
| **Numerical** | NumPy, SciPy |
| **Visualization** | matplotlib, seaborn |
| **Data** | pandas, joblib |
| **Storage** | Filesystem (images, models, results) |

---

**This architecture provides a solid foundation for plant image classification while remaining flexible for future enhancements.**