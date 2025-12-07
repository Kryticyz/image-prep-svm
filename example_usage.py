#!/usr/bin/env python3
"""
Example Usage Script for SVM Plant Classification

This script demonstrates how to use the SVM plant classifier for
identifying plant species in images. Follow along with the comments
to understand each step.

Author: SVM Plant Classifier
Date: 2024
"""

import os
import sys

# Add the src directory to Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import our custom modules
from data_loader import PlantImageDataLoader
from feature_extractor import ImageFeatureExtractor
from train_svm import SVMPlantClassifier, save_results, visualize_results


def example_1_organize_data():
    """
    Example 1: Organizing Your Image Dataset

    Before training, you need to organize your images into the correct structure.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Organizing Image Dataset")
    print("=" * 70)

    # Initialize the data loader
    data_loader = PlantImageDataLoader(data_root="data")

    # Option A: Organize images from a source directory
    # This will split your images into train/test sets automatically
    source_directory = "src/Myosotis_sylvatica"  # Replace with your path

    if os.path.exists(source_directory):
        print(f"\nOrganizing images from: {source_directory}")
        data_loader.organize_data_from_source(
            source_dir=source_directory,
            test_size=0.2,  # 20% for testing, 80% for training
            random_state=42,  # For reproducibility
            class_label="target_species",  # This is the target species
        )
        print("✓ Images organized successfully!")
    else:
        print(f"\nSource directory not found: {source_directory}")
        print("You can create a synthetic dataset for testing instead.")

    # Option B: Create a synthetic dataset for testing/learning
    print("\nCreating synthetic dataset for demonstration...")
    data_loader.create_synthetic_dataset(n_samples_per_class=50)

    # Show dataset summary
    data_loader.print_dataset_summary()


def example_2_train_basic_model():
    """
    Example 2: Training a Basic SVM Model

    Train a simple model with default parameters (fast but may not be optimal).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Training a Basic Model (No Grid Search)")
    print("=" * 70)

    # Configuration with grid search disabled for faster training
    config = {
        "image_size": (128, 128),  # Smaller size for faster processing
        "use_color_histogram": True,
        "use_hog": True,
        "use_lbp": False,  # Disable LBP for faster training
        "use_statistics": True,
        "perform_grid_search": False,  # Skip grid search for speed
        "cross_validation_folds": 3,
        "random_state": 42,
    }

    # Initialize classifier
    classifier = SVMPlantClassifier(config=config)
    feature_extractor = classifier.build_feature_extractor()

    # Load data
    data_loader = PlantImageDataLoader(data_root="data")

    print("\nLoading training data...")
    X_train, y_train, train_files = data_loader.load_dataset(
        feature_extractor, split="train"
    )

    print("\nLoading test data...")
    X_test, y_test, test_files = data_loader.load_dataset(
        feature_extractor, split="test"
    )

    # Train the model
    print("\nTraining model...")
    training_history = classifier.train(X_train, y_train)

    # Evaluate the model
    print("\nEvaluating model...")
    results = classifier.evaluate(X_test, y_test)

    # Save the model
    model_path = classifier.save_model(save_dir="models", model_name="basic_svm_model")

    print(f"\n✓ Model saved to: {model_path}")

    return classifier, model_path


def example_3_train_optimized_model():
    """
    Example 3: Training an Optimized Model with Grid Search

    Train a model with hyperparameter tuning for best performance.
    This takes longer but usually gives better results.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Training an Optimized Model (With Grid Search)")
    print("=" * 70)

    # Configuration with all features enabled and grid search
    config = {
        "image_size": (224, 224),  # Higher resolution for better features
        "use_color_histogram": True,
        "use_hog": True,
        "use_lbp": True,
        "use_statistics": True,
        "perform_grid_search": True,  # Enable hyperparameter tuning
        "cross_validation_folds": 5,
        "random_state": 42,
    }

    # Initialize classifier
    classifier = SVMPlantClassifier(config=config)
    feature_extractor = classifier.build_feature_extractor()

    # Load data
    data_loader = PlantImageDataLoader(data_root="data")

    print("\nLoading data...")
    X_train, y_train, train_files = data_loader.load_dataset(
        feature_extractor, split="train"
    )
    X_test, y_test, test_files = data_loader.load_dataset(
        feature_extractor, split="test"
    )

    # Train with grid search (this may take several minutes)
    print("\nTraining model with grid search...")
    print("(This may take several minutes...)")
    training_history = classifier.train(X_train, y_train)

    # Evaluate
    results = classifier.evaluate(X_test, y_test)

    # Save model and results
    model_path = classifier.save_model(
        save_dir="models", model_name="optimized_svm_model"
    )

    # Save results and create visualizations
    save_results(results, training_history, save_dir="results")
    visualize_results(results, save_dir="results")

    print(f"\n✓ Model and results saved!")

    return classifier, model_path


def example_4_single_prediction(model_path):
    """
    Example 4: Predicting a Single Image

    Use a trained model to classify a single plant image.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Single Image Prediction")
    print("=" * 70)

    # Load the trained model
    classifier = SVMPlantClassifier()
    classifier.load_model(model_path)

    # Get a test image
    data_loader = PlantImageDataLoader(data_root="data")
    test_images = data_loader._get_image_files("data/test/target_species")

    if len(test_images) > 0:
        test_image = test_images[0]
        print(f"\nPredicting image: {test_image}")

        # Make prediction
        result = classifier.predict(test_image)

        # Display results
        print("\n" + "-" * 50)
        print("PREDICTION RESULTS")
        print("-" * 50)
        print(f"Class: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nProbabilities:")
        print(f"  Target Species: {result['probability']['target_species']:.4f}")
        print(f"  Other Species:  {result['probability']['other_species']:.4f}")
        print("-" * 50)
    else:
        print("\nNo test images found!")


def example_5_batch_prediction(model_path):
    """
    Example 5: Batch Prediction on Multiple Images

    Classify multiple images at once and get statistics.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Batch Prediction")
    print("=" * 70)

    # Load the trained model
    classifier = SVMPlantClassifier()
    classifier.load_model(model_path)

    # Get all test images
    data_loader = PlantImageDataLoader(data_root="data")
    target_images = data_loader._get_image_files("data/test/target_species")
    other_images = data_loader._get_image_files("data/test/other_species")

    all_test_images = target_images[:5] + other_images[:5]  # Sample 10 images

    if len(all_test_images) > 0:
        print(f"\nPredicting {len(all_test_images)} images...")

        # Track predictions
        correct = 0
        total = len(all_test_images)

        for image_path in all_test_images:
            # Determine true label from directory
            true_label = 1 if "target_species" in image_path else 0

            # Predict
            result = classifier.predict(image_path)

            # Check if correct
            is_correct = result["prediction"] == true_label
            correct += is_correct

            # Print result
            status = "✓" if is_correct else "✗"
            print(
                f"{status} {os.path.basename(image_path)}: "
                f"{result['class_name']} ({result['confidence']:.1%})"
            )

        # Summary
        accuracy = correct / total
        print(f"\n{'=' * 50}")
        print(f"Accuracy: {correct}/{total} ({accuracy:.1%})")
        print(f"{'=' * 50}")
    else:
        print("\nNo test images found!")


def example_6_custom_features():
    """
    Example 6: Training with Custom Feature Configuration

    Experiment with different feature combinations to see what works best.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Custom Feature Configuration")
    print("=" * 70)

    # Try different feature combinations
    feature_configs = [
        {
            "name": "Color Only",
            "use_color_histogram": True,
            "use_hog": False,
            "use_lbp": False,
            "use_statistics": False,
        },
        {
            "name": "HOG Only",
            "use_color_histogram": False,
            "use_hog": True,
            "use_lbp": False,
            "use_statistics": False,
        },
        {
            "name": "Color + Statistics",
            "use_color_histogram": True,
            "use_hog": False,
            "use_lbp": False,
            "use_statistics": True,
        },
    ]

    data_loader = PlantImageDataLoader(data_root="data")

    for feat_config in feature_configs:
        print(f"\n{'=' * 50}")
        print(f"Testing: {feat_config['name']}")
        print(f"{'=' * 50}")

        # Create config
        config = {
            "image_size": (128, 128),
            "use_color_histogram": feat_config["use_color_histogram"],
            "use_hog": feat_config["use_hog"],
            "use_lbp": feat_config["use_lbp"],
            "use_statistics": feat_config["use_statistics"],
            "perform_grid_search": False,
            "cross_validation_folds": 3,
            "random_state": 42,
        }

        # Train and evaluate
        classifier = SVMPlantClassifier(config=config)
        feature_extractor = classifier.build_feature_extractor()

        X_train, y_train, _ = data_loader.load_dataset(feature_extractor, split="train")
        X_test, y_test, _ = data_loader.load_dataset(feature_extractor, split="test")

        classifier.train(X_train, y_train)
        results = classifier.evaluate(X_test, y_test)

        print(f"\nResults for {feat_config['name']}:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1_score']:.4f}")


def main():
    """
    Main function to run all examples.

    Uncomment the examples you want to run.
    """
    print("\n" + "=" * 70)
    print("SVM PLANT CLASSIFICATION - EXAMPLE USAGE")
    print("=" * 70)

    # Example 1: Organize your data
    print("\nRunning Example 1: Organizing Data")
    example_1_organize_data()

    # Example 2: Train a basic model (fast)
    print("\nRunning Example 2: Training Basic Model")
    classifier, model_path = example_2_train_basic_model()

    # Example 4: Single prediction
    print("\nRunning Example 4: Single Prediction")
    example_4_single_prediction(model_path)

    # Example 5: Batch prediction
    print("\nRunning Example 5: Batch Prediction")
    example_5_batch_prediction(model_path)

    # Example 3: Train optimized model (slow but better)
    # Uncomment to run (this takes longer)
    # print("\nRunning Example 3: Training Optimized Model")
    # classifier, model_path = example_3_train_optimized_model()

    # Example 6: Experiment with features
    # Uncomment to run feature comparison
    # print("\nRunning Example 6: Custom Feature Configuration")
    # example_6_custom_features()

    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review the results in the 'results/' directory")
    print("  2. Try uncommenting Example 3 for better performance")
    print("  3. Experiment with Example 6 to find optimal features")
    print("  4. Add your own plant images to improve the model")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
