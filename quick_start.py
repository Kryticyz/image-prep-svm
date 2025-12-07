#!/usr/bin/env python3
"""
Quick Start Script for SVM Plant Classification

This script demonstrates the complete workflow for training and using
an SVM-based plant species classifier.
"""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import PlantImageDataLoader
from feature_extractor import ImageFeatureExtractor
from train_svm import SVMPlantClassifier, save_results, visualize_results


def quick_start():
    """
    Quick start guide for SVM plant classification.
    """

    print("=" * 70)
    print(" SVM PLANT CLASSIFICATION - QUICK START")
    print("=" * 70)

    print("\nThis script will guide you through:")
    print("  1. Dataset preparation")
    print("  2. Model training")
    print("  3. Model evaluation")
    print("  4. Making predictions")

    # Check if data exists
    data_root = "data"
    data_loader = PlantImageDataLoader(data_root=data_root)
    stats = data_loader.get_dataset_statistics()

    # Step 1: Prepare dataset
    print("\n" + "=" * 70)
    print("STEP 1: DATASET PREPARATION")
    print("=" * 70)

    if stats["train"]["total"] == 0 and stats["test"]["total"] == 0:
        print("\n⚠️  No dataset found!")
        print("\nOptions:")
        print("  1. Create a synthetic dataset for testing")
        print("  2. Organize existing images (if you have Myosotis_sylvatica images)")
        print("  3. Exit and manually add images to data/ directory")

        choice = input("\nEnter your choice (1, 2, or 3): ").strip()

        if choice == "1":
            print("\n📊 Creating synthetic dataset...")
            data_loader.create_synthetic_dataset(n_samples_per_class=100)
            print("✓ Synthetic dataset created!")

        elif choice == "2":
            myosotis_dir = "src/Myosotis_sylvatica"
            if os.path.exists(myosotis_dir):
                print(f"\n📂 Found images in {myosotis_dir}")
                print("Organizing as target species...")
                data_loader.organize_data_from_source(
                    source_dir=myosotis_dir,
                    test_size=0.2,
                    random_state=42,
                    class_label="target_species",
                )
                print("✓ Data organized!")
                print("\n⚠️  Note: You still need to add 'other_species' images!")
                print("   Please add images to data/train/other_species/")
                print("   and data/test/other_species/")

                # Create synthetic other species
                print("\nCreating synthetic 'other species' for demonstration...")
                import numpy as np
                from PIL import Image

                os.makedirs("data/train/other_species", exist_ok=True)
                os.makedirs("data/test/other_species", exist_ok=True)

                for split, n_samples in [("train", 80), ("test", 20)]:
                    split_dir = os.path.join(data_root, split, "other_species")
                    for i in range(n_samples):
                        img_array = np.random.randint(
                            0, 255, (224, 224, 3), dtype=np.uint8
                        )
                        img_array[:, :, 0] = np.clip(img_array[:, :, 0] + 50, 0, 255)
                        img = Image.fromarray(img_array)
                        img.save(
                            os.path.join(split_dir, f"synthetic_other_{i:04d}.png")
                        )

                print("✓ Synthetic other species created!")
            else:
                print(f"\n✗ Directory not found: {myosotis_dir}")
                print("Please manually add images or choose option 1")
                return

        else:
            print("\nPlease add images manually to:")
            print("  - data/train/target_species/")
            print("  - data/train/other_species/")
            print("  - data/test/target_species/")
            print("  - data/test/other_species/")
            return
    else:
        print("\n✓ Dataset found!")
        data_loader.print_dataset_summary()

    # Check if we have both classes
    stats = data_loader.get_dataset_statistics()
    if stats["train"]["target_species"] == 0 or stats["train"]["other_species"] == 0:
        print("\n⚠️  Warning: Dataset is incomplete!")
        print("You need images for both 'target_species' and 'other_species'")
        return

    # Step 2: Configure and train model
    print("\n" + "=" * 70)
    print("STEP 2: MODEL TRAINING")
    print("=" * 70)

    print("\nConfiguration:")
    config = {
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

    for key, value in config.items():
        print(f"  {key}: {value}")

    # Ask if user wants to skip grid search (faster training)
    print("\n⏱️  Grid search can take several minutes.")
    skip_gs = input("Skip grid search for faster training? (yes/no): ").strip().lower()
    if skip_gs == "yes":
        config["perform_grid_search"] = False
        print("Grid search disabled - using default parameters")

    # Initialize classifier
    print("\n🔧 Initializing classifier...")
    classifier = SVMPlantClassifier(config=config)
    feature_extractor = classifier.build_feature_extractor()

    # Load data
    print("\n📁 Loading training data...")
    X_train, y_train, train_files = data_loader.load_dataset(
        feature_extractor, split="train"
    )

    print("\n📁 Loading test data...")
    X_test, y_test, test_files = data_loader.load_dataset(
        feature_extractor, split="test"
    )

    # Train model
    print("\n🚀 Training model...")
    training_history = classifier.train(X_train, y_train)

    # Step 3: Evaluate model
    print("\n" + "=" * 70)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 70)

    results = classifier.evaluate(X_test, y_test)

    # Save model
    print("\n💾 Saving model...")
    os.makedirs("models", exist_ok=True)
    model_path = classifier.save_model(
        save_dir="models", model_name="quick_start_model"
    )

    # Save results
    print("\n💾 Saving results...")
    os.makedirs("results", exist_ok=True)
    save_results(results, training_history, save_dir="results")

    # Create visualizations
    print("\n📊 Creating visualizations...")
    visualize_results(results, save_dir="results")

    # Step 4: Example predictions
    print("\n" + "=" * 70)
    print("STEP 4: MAKING PREDICTIONS")
    print("=" * 70)

    print("\n🔮 Testing predictions on a few test images...")

    # Make predictions on first few test images
    sample_size = min(5, len(test_files))
    for i in range(sample_size):
        image_path = test_files[i]
        true_label = y_test[i]

        result = classifier.predict(image_path)

        correct = "✓" if result["prediction"] == true_label else "✗"
        print(f"\n{correct} Image: {os.path.basename(image_path)}")
        print(
            f"  True Label: {'Target Species' if true_label == 1 else 'Other Species'}"
        )
        print(f"  Predicted: {result['class_name']}")
        print(f"  Confidence: {result['confidence']:.2%}")

    # Summary
    print("\n" + "=" * 70)
    print("QUICK START COMPLETE!")
    print("=" * 70)

    print("\n📁 Files created:")
    print(f"  ✓ Model: {model_path}")
    print(f"  ✓ Results: results/")
    print(f"  ✓ Visualizations: results/")

    print("\n📚 Next steps:")
    print("  1. Check results/ directory for visualizations and metrics")
    print("  2. Use the trained model for predictions:")
    print(f"     python src/predict.py --model {model_path} --images your_image.jpg")
    print("  3. Add more training data to improve accuracy")
    print("  4. Adjust hyperparameters in the config")

    print("\n💡 Tips:")
    print("  - More training data = better performance")
    print("  - Balance your classes (similar number of images per class)")
    print("  - Use diverse images (different angles, lighting, backgrounds)")
    print("  - Check confusion matrix to understand errors")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        quick_start()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
