"""
Prediction Script for Plant Image Classification

This script uses a trained SVM model to classify plant images
as either the target species or other species.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class PlantImagePredictor:
    """
    Predictor for plant image classification using trained SVM model.
    """

    def __init__(self, model_path):
        """
        Initialize the predictor with a trained model.

        Args:
            model_path (str): Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_extractor = None
        self.config = None

        self._load_model()

    def _load_model(self):
        """Load the trained model and associated objects."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        print(f"Loading model from: {self.model_path}")
        model_data = joblib.load(self.model_path)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_extractor = model_data["feature_extractor"]
        self.config = model_data.get("config", {})

        print("Model loaded successfully!")
        print(f"Feature dimension: {self.feature_extractor.get_feature_dimension()}")

    def predict_single(self, image_path, verbose=True):
        """
        Predict the class of a single image.

        Args:
            image_path (str): Path to the image file
            verbose (bool): Whether to print prediction details

        Returns:
            dict: Prediction results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Extract features
        features = self.feature_extractor.extract_features(image_path)
        features = features.reshape(1, -1)

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]

        # Prepare results
        result = {
            "image_path": image_path,
            "prediction": int(prediction),
            "class_name": "Target Species" if prediction == 1 else "Other Species",
            "probability": {
                "other_species": float(probability[0]),
                "target_species": float(probability[1]),
            },
            "confidence": float(max(probability)),
        }

        if verbose:
            self._print_prediction(result)

        return result

    def predict_batch(self, image_paths, verbose=True):
        """
        Predict classes for multiple images.

        Args:
            image_paths (list): List of image file paths
            verbose (bool): Whether to print prediction details

        Returns:
            list: List of prediction results
        """
        results = []

        print(f"\nProcessing {len(image_paths)} images...")

        for i, image_path in enumerate(image_paths, 1):
            try:
                if verbose:
                    print(f"\n[{i}/{len(image_paths)}] Processing: {image_path}")

                result = self.predict_single(image_path, verbose=verbose)
                results.append(result)

            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append(
                    {
                        "image_path": image_path,
                        "error": str(e),
                        "prediction": None,
                    }
                )

        return results

    def predict_directory(self, directory_path, recursive=True, verbose=True):
        """
        Predict classes for all images in a directory.

        Args:
            directory_path (str): Path to directory containing images
            recursive (bool): Whether to search subdirectories
            verbose (bool): Whether to print prediction details

        Returns:
            list: List of prediction results
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Get all image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        image_paths = []

        if recursive:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(directory_path, file))

        if len(image_paths) == 0:
            print(f"No images found in {directory_path}")
            return []

        print(f"Found {len(image_paths)} images in {directory_path}")

        return self.predict_batch(image_paths, verbose=verbose)

    def _print_prediction(self, result):
        """
        Print prediction result in a formatted way.

        Args:
            result (dict): Prediction result dictionary
        """
        print("-" * 60)
        print(f"Image: {os.path.basename(result['image_path'])}")
        print(f"Prediction: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities:")
        print(
            f"  Other Species:  {result['probability']['other_species']:.4f} ({result['probability']['other_species'] * 100:.2f}%)"
        )
        print(
            f"  Target Species: {result['probability']['target_species']:.4f} ({result['probability']['target_species'] * 100:.2f}%)"
        )
        print("-" * 60)

    def visualize_prediction(self, result, save_path=None):
        """
        Visualize a single prediction with the image.

        Args:
            result (dict): Prediction result
            save_path (str): Path to save the visualization (optional)
        """
        # Load image
        image = Image.open(result["image_path"])

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Display image
        ax1.imshow(image)
        ax1.axis("off")
        ax1.set_title(f"Input Image\n{os.path.basename(result['image_path'])}")

        # Display prediction as bar chart
        classes = ["Other Species", "Target Species"]
        probabilities = [
            result["probability"]["other_species"],
            result["probability"]["target_species"],
        ]
        colors = [
            "red" if result["prediction"] == 0 else "lightcoral",
            "lightgreen" if result["prediction"] == 1 else "green",
        ]

        bars = ax2.barh(classes, probabilities, color=colors)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("Probability")
        ax2.set_title(
            f"Prediction: {result['class_name']}\nConfidence: {result['confidence']:.2%}"
        )

        # Add probability labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            ax2.text(
                prob + 0.02,
                i,
                f"{prob:.2%}",
                va="center",
                fontweight="bold",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_summary_report(self, results, save_path=None):
        """
        Generate a summary report of batch predictions.

        Args:
            results (list): List of prediction results
            save_path (str): Path to save the report (optional)
        """
        # Filter out errors
        valid_results = [r for r in results if r.get("prediction") is not None]

        if len(valid_results) == 0:
            print("No valid predictions to summarize.")
            return

        # Calculate statistics
        total = len(valid_results)
        target_count = sum(1 for r in valid_results if r["prediction"] == 1)
        other_count = total - target_count

        avg_confidence = np.mean([r["confidence"] for r in valid_results])
        avg_target_prob = np.mean(
            [r["probability"]["target_species"] for r in valid_results]
        )

        # Create summary
        summary = {
            "total_images": total,
            "target_species_count": target_count,
            "other_species_count": other_count,
            "target_species_percentage": (target_count / total * 100)
            if total > 0
            else 0,
            "average_confidence": avg_confidence,
            "average_target_probability": avg_target_prob,
        }

        # Print summary
        print("\n" + "=" * 60)
        print("PREDICTION SUMMARY")
        print("=" * 60)
        print(f"Total Images Processed: {total}")
        print(f"Target Species: {target_count} ({target_count / total * 100:.1f}%)")
        print(f"Other Species: {other_count} ({other_count / total * 100:.1f}%)")
        print(f"Average Confidence: {avg_confidence:.2%}")
        print(f"Average Target Species Probability: {avg_target_prob:.2%}")
        print("=" * 60)

        # Save to file if requested
        if save_path:
            # Include detailed results
            summary["detailed_results"] = valid_results

            with open(save_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"\nDetailed report saved to: {save_path}")

        return summary


def get_image_paths_from_args(args):
    """
    Get list of image paths from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        list: List of image file paths
    """
    image_paths = []

    # From individual files
    if args.images:
        image_paths.extend(args.images)

    # From directory
    if args.directory:
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

        if args.recursive:
            for root, dirs, files in os.walk(args.directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(args.directory):
                file_path = os.path.join(args.directory, file)
                if os.path.isfile(file_path) and any(
                    file.lower().endswith(ext) for ext in image_extensions
                ):
                    image_paths.append(file_path)

    return image_paths


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description="Predict plant species using trained SVM model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model file (.pkl)",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        help="Path(s) to image file(s) to classify",
    )
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to directory containing images to classify",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directory recursively for images",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save prediction results (JSON format)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations of predictions",
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default="predictions",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.images and not args.directory:
        parser.error("Must specify either --images or --directory")

    # Initialize predictor
    try:
        predictor = PlantImagePredictor(args.model)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

    # Get image paths
    image_paths = get_image_paths_from_args(args)

    if len(image_paths) == 0:
        print("No images found to process.")
        sys.exit(1)

    print(f"\nFound {len(image_paths)} image(s) to process.")

    # Make predictions
    verbose = not args.quiet
    results = predictor.predict_batch(image_paths, verbose=verbose)

    # Generate summary
    summary = predictor.generate_summary_report(
        results, save_path=args.output if args.output else None
    )

    # Create visualizations if requested
    if args.visualize:
        os.makedirs(args.viz_dir, exist_ok=True)
        print(f"\nCreating visualizations in {args.viz_dir}...")

        for i, result in enumerate(results):
            if result.get("prediction") is not None:
                viz_filename = f"prediction_{i:04d}_{os.path.splitext(os.path.basename(result['image_path']))[0]}.png"
                viz_path = os.path.join(args.viz_dir, viz_filename)

                try:
                    predictor.visualize_prediction(result, save_path=viz_path)
                except Exception as e:
                    print(
                        f"Error creating visualization for {result['image_path']}: {str(e)}"
                    )

        print(f"Visualizations saved to: {args.viz_dir}")

    print("\nPrediction complete!")


if __name__ == "__main__":
    main()
