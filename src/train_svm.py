"""
SVM Training Script for Plant Image Classification

This script trains a Support Vector Machine (SVM) to classify plant images
as either the target species or other species.
"""

import argparse
import json
import os
import pickle
import time
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_loader import ImageDataLoader
from feature_extractor import ImageFeatureExtractor


class SVMPlantClassifier:
    """
    SVM-based plant species classifier.
    """

    def __init__(self, config=None):
        """
        Initialize the classifier.

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.feature_extractor = None
        self.scaler = None
        self.model = None
        self.training_history = {}

    def _get_default_config(self):
        """Get default configuration."""
        return {
            "image_size": (224, 224),
            "use_color_histogram": True,
            "use_hog": True,
            "use_lbp": True,
            "use_statistics": True,
            "svm_kernel": "rbf",
            "svm_C": 1.0,
            "svm_gamma": "scale",
            "perform_grid_search": True,
            "cross_validation_folds": 5,
            "random_state": 42,
        }

    def build_feature_extractor(self):
        """Build the feature extractor."""
        self.feature_extractor = ImageFeatureExtractor(
            image_size=tuple(self.config["image_size"]),
            use_color_histogram=self.config["use_color_histogram"],
            use_hog=self.config["use_hog"],
            use_lbp=self.config["use_lbp"],
            use_statistics=self.config["use_statistics"],
        )

        feature_dim = self.feature_extractor.get_feature_dimension()
        print(f"\nFeature extractor built. Total feature dimension: {feature_dim}")

        return self.feature_extractor

    def train(self, X_train, y_train):
        """
        Train the SVM classifier.

        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels

        Returns:
            dict: Training results
        """
        print("\n" + "=" * 60)
        print("TRAINING SVM CLASSIFIER")
        print("=" * 60)

        start_time = time.time()

        # Scale features
        print("\nScaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Perform grid search for hyperparameter tuning
        if self.config["perform_grid_search"]:
            print("\nPerforming grid search for hyperparameter tuning...")
            self.model = self._grid_search(X_train_scaled, y_train)
        else:
            print("\nTraining SVM with default parameters...")
            self.model = SVC(
                kernel=self.config["svm_kernel"],
                C=self.config["svm_C"],
                gamma=self.config["svm_gamma"],
                probability=True,
                random_state=self.config["random_state"],
            )
            self.model.fit(X_train_scaled, y_train)

        training_time = time.time() - start_time

        # Cross-validation
        print("\nPerforming cross-validation...")
        cv_scores = cross_val_score(
            self.model,
            X_train_scaled,
            y_train,
            cv=self.config["cross_validation_folds"],
            scoring="accuracy",
        )

        # Training set performance
        y_train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Store training history
        self.training_history = {
            "training_time": training_time,
            "train_accuracy": train_accuracy,
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
            "best_params": self.model.get_params(),
        }

        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(
            f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
        )

        return self.training_history

    def _grid_search(self, X_train, y_train):
        """
        Perform grid search for hyperparameter tuning.

        Args:
            X_train (numpy.ndarray): Training features (scaled)
            y_train (numpy.ndarray): Training labels

        Returns:
            SVC: Best SVM model
        """
        # Define parameter grid
        param_grid = {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "kernel": ["rbf", "linear"],
        }

        # Create grid search
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=self.config["random_state"]),
            param_grid,
            cv=self.config["cross_validation_folds"],
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )

        # Fit grid search
        grid_search.fit(X_train, y_train)

        print(f"\nBest parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model.

        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels

        Returns:
            dict: Evaluation results
        """
        print("\n" + "=" * 60)
        print("EVALUATING MODEL")
        print("=" * 60)

        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)

        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        class_report = classification_report(
            y_test, y_pred, target_names=["Other Species", "Target Species"]
        )

        # Store results
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "y_pred_proba": y_pred_proba.tolist(),
        }

        # Print results
        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")

        print(f"\nConfusion Matrix:")
        print(cm)

        print(f"\nClassification Report:")
        print(class_report)

        return results

    def save_model(self, save_dir="models", model_name=None):
        """
        Save the trained model and associated objects.

        Args:
            save_dir (str): Directory to save the model
            model_name (str): Name for the model file
        """
        os.makedirs(save_dir, exist_ok=True)

        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"svm_plant_classifier_{timestamp}"

        model_path = os.path.join(save_dir, f"{model_name}.pkl")

        # Save model, scaler, and feature extractor
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_extractor": self.feature_extractor,
            "config": self.config,
            "training_history": self.training_history,
        }

        joblib.dump(model_data, model_path)

        print(f"\nModel saved to: {model_path}")

        return model_path

    def load_model(self, model_path):
        """
        Load a trained model.

        Args:
            model_path (str): Path to the saved model
        """
        model_data = joblib.load(model_path)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_extractor = model_data["feature_extractor"]
        self.config = model_data["config"]
        self.training_history = model_data.get("training_history", {})

        print(f"Model loaded from: {model_path}")

    def predict(self, image_path):
        """
        Predict the class of a single image.

        Args:
            image_path (str): Path to the image

        Returns:
            dict: Prediction results
        """
        # Extract features
        features = self.feature_extractor.extract_features(image_path)
        features = features.reshape(1, -1)

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]

        result = {
            "prediction": int(prediction),
            "class_name": "Target Species" if prediction == 1 else "Other Species",
            "probability": {
                "other_species": float(probability[0]),
                "target_species": float(probability[1]),
            },
            "confidence": float(max(probability)),
        }

        return result


def visualize_results(results, save_dir="results"):
    """
    Create visualizations of the model results.

    Args:
        results (dict): Evaluation results
        save_dir (str): Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = np.array(results["confusion_matrix"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Other Species", "Target Species"],
        yticklabels=["Other Species", "Target Species"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    y_true = np.array(results["y_true"])
    y_pred_proba = np.array(results["y_pred_proba"])

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = results["roc_auc"]

    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300)
    plt.close()

    # 3. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "precision_recall_curve.png"), dpi=300)
    plt.close()

    # 4. Prediction Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(
        y_pred_proba[y_true == 0],
        bins=30,
        alpha=0.5,
        label="Other Species",
        color="red",
    )
    plt.hist(
        y_pred_proba[y_true == 1],
        bins=30,
        alpha=0.5,
        label="Target Species",
        color="green",
    )
    plt.xlabel("Predicted Probability (Target Species)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predicted Probabilities")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prediction_distribution.png"), dpi=300)
    plt.close()

    print(f"\nVisualizations saved to: {save_dir}")


def save_results(results, training_history, save_dir="results"):
    """
    Save results to JSON file.

    Args:
        results (dict): Evaluation results
        training_history (dict): Training history
        save_dir (str): Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_dir, f"results_{timestamp}.json")

    all_results = {
        "timestamp": timestamp,
        "training": training_history,
        "evaluation": {
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1_score": results["f1_score"],
            "roc_auc": results["roc_auc"],
            "confusion_matrix": results["confusion_matrix"],
        },
    }

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to: {results_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train SVM for plant image classification"
    )
    parser.add_argument(
        "--data-root", type=str, default="data", help="Root directory for data"
    )
    parser.add_argument(
        "--model-dir", type=str, default="models", help="Directory to save models"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Image size (width height)",
    )
    parser.add_argument(
        "--no-grid-search",
        action="store_true",
        help="Skip grid search for hyperparameters",
    )
    parser.add_argument(
        "--create-synthetic",
        action="store_true",
        help="Create synthetic dataset for testing",
    )

    args = parser.parse_args()

    # Configuration
    config = {
        "image_size": tuple(args.image_size),
        "use_color_histogram": True,
        "use_hog": True,
        "use_lbp": True,
        "use_statistics": True,
        "svm_kernel": "rbf",
        "svm_C": 1.0,
        "svm_gamma": "scale",
        "perform_grid_search": not args.no_grid_search,
        "cross_validation_folds": 5,
        "random_state": 42,
    }

    print("\n" + "=" * 60)
    print("SVM BINARY IMAGE CLASSIFIER - TRAINING")
    print("=" * 60)

    # Initialize data loader
    data_loader = ImageDataLoader(data_root=args.data_root)

    # Create synthetic dataset if requested
    if args.create_synthetic:
        data_loader.create_synthetic_dataset(n_samples_per_class=100)
    else:
        data_loader.print_dataset_summary()

    # Initialize classifier
    classifier = SVMPlantClassifier(config=config)

    # Build feature extractor
    feature_extractor = classifier.build_feature_extractor()

    # Load training data
    print("\n" + "=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)
    X_train, y_train, train_files = data_loader.load_dataset(
        feature_extractor, split="train"
    )

    # Load test data
    print("\n" + "=" * 60)
    print("LOADING TEST DATA")
    print("=" * 60)
    X_test, y_test, test_files = data_loader.load_dataset(
        feature_extractor, split="test"
    )

    # Train model
    training_history = classifier.train(X_train, y_train)

    # Evaluate model
    results = classifier.evaluate(X_test, y_test)

    # Save model
    model_path = classifier.save_model(save_dir=args.model_dir)

    # Save results
    save_results(results, training_history, save_dir=args.results_dir)

    # Create visualizations
    visualize_results(results, save_dir=args.results_dir)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: {model_path}")
    print(f"Results saved to: {args.results_dir}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
