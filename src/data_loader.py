"""
Data Loader Module for Image Classification

This module handles loading, preprocessing, and managing image datasets
for SVM-based image classification.
"""

import os
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class ImageDataLoader:
    """
    Manages loading and organizing image datasets for classification.
    """

    def __init__(self, data_root="data"):
        """
        Initialize the data loader.

        Args:
            data_root (str): Root directory for data storage
        """
        self.data_root = data_root
        self.train_dir = os.path.join(data_root, "train")
        self.test_dir = os.path.join(data_root, "test")

        # Class names
        self.target_class = "target_species"
        self.other_class = "other_species"

        # Image extensions to consider
        self.image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

    def load_dataset(self, feature_extractor, split="train"):
        """
        Load and extract features from a dataset split.

        Args:
            feature_extractor: ImageFeatureExtractor instance
            split (str): 'train' or 'test'

        Returns:
            tuple: (X, y, file_paths) where X is features, y is labels,
                   and file_paths is list of image paths
        """
        if split == "train":
            data_dir = self.train_dir
        elif split == "test":
            data_dir = self.test_dir
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")

        # Load target species (positive class)
        target_dir = os.path.join(data_dir, self.target_class)
        X_target, files_target = self._extract_features_from_class(
            target_dir, feature_extractor
        )
        y_target = np.ones(len(X_target), dtype=int)  # Label 1 for target species

        # Load other species (negative class)
        other_dir = os.path.join(data_dir, self.other_class)
        X_other, files_other = self._extract_features_from_class(
            other_dir, feature_extractor
        )
        y_other = np.zeros(len(X_other), dtype=int)  # Label 0 for other species

        # Combine datasets
        X = np.vstack([X_target, X_other]) if len(X_other) > 0 else X_target
        y = np.concatenate([y_target, y_other]) if len(X_other) > 0 else y_target
        file_paths = files_target + files_other

        print(f"\n{split.capitalize()} dataset loaded:")
        print(f"  Target species: {len(y_target)} images")
        print(f"  Other species: {len(y_other)} images")
        print(f"  Total: {len(y)} images")
        print(f"  Feature dimension: {X.shape[1]}")

        return X, y, file_paths

    def _extract_features_from_class(self, class_dir, feature_extractor):
        """
        Extract features from all images in a class directory.

        Args:
            class_dir (str): Path to class directory
            feature_extractor: ImageFeatureExtractor instance

        Returns:
            tuple: (features_array, file_paths)
        """
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            return np.array([]), []

        # Get all image files
        image_files = self._get_image_files(class_dir)

        if len(image_files) == 0:
            print(f"Warning: No images found in {class_dir}")
            return np.array([]), []

        # Extract features
        features_list = []
        valid_files = []

        class_name = os.path.basename(class_dir)
        print(f"\nExtracting features from {class_name} ({len(image_files)} images)...")

        for image_path in tqdm(image_files):
            try:
                features = feature_extractor.extract_features(image_path)
                features_list.append(features)
                valid_files.append(image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue

        # Convert to numpy array
        features_array = np.array(features_list) if features_list else np.array([])

        return features_array, valid_files

    def _get_image_files(self, directory):
        """
        Get all image file paths from a directory.

        Args:
            directory (str): Directory path

        Returns:
            list: List of image file paths
        """
        image_files = []

        if not os.path.exists(directory):
            return image_files

        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.image_extensions):
                    image_files.append(os.path.join(root, file))

        return sorted(image_files)

    def organize_data_from_source(
        self, source_dir, test_size=0.2, random_state=42, class_label="target_species"
    ):
        """
        Organize images from a source directory into train/test splits.

        Args:
            source_dir (str): Source directory containing images
            test_size (float): Proportion of data for testing (0.0 to 1.0)
            random_state (int): Random seed for reproducibility
            class_label (str): 'target_species' or 'other_species'
        """
        # Get all image files
        image_files = self._get_image_files(source_dir)

        if len(image_files) == 0:
            print(f"No images found in {source_dir}")
            return

        print(f"Found {len(image_files)} images in {source_dir}")

        # Split into train and test
        train_files, test_files = train_test_split(
            image_files, test_size=test_size, random_state=random_state
        )

        # Determine destination directories
        train_dest = os.path.join(self.train_dir, class_label)
        test_dest = os.path.join(self.test_dir, class_label)

        # Create directories
        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(test_dest, exist_ok=True)

        # Copy files
        print(f"Copying {len(train_files)} images to training set...")
        self._copy_files(train_files, train_dest)

        print(f"Copying {len(test_files)} images to test set...")
        self._copy_files(test_files, test_dest)

        print("Data organization complete!")

    def _copy_files(self, file_list, destination_dir):
        """
        Copy files to a destination directory.

        Args:
            file_list (list): List of file paths to copy
            destination_dir (str): Destination directory
        """
        for file_path in tqdm(file_list):
            filename = os.path.basename(file_path)
            dest_path = os.path.join(destination_dir, filename)

            # Handle duplicate filenames
            counter = 1
            base_name, ext = os.path.splitext(filename)
            while os.path.exists(dest_path):
                new_filename = f"{base_name}_{counter}{ext}"
                dest_path = os.path.join(destination_dir, new_filename)
                counter += 1

            shutil.copy2(file_path, dest_path)

    def get_dataset_statistics(self):
        """
        Get statistics about the current dataset.

        Returns:
            dict: Dataset statistics
        """
        stats = {
            "train": {
                "target_species": len(
                    self._get_image_files(
                        os.path.join(self.train_dir, self.target_class)
                    )
                ),
                "other_species": len(
                    self._get_image_files(
                        os.path.join(self.train_dir, self.other_class)
                    )
                ),
            },
            "test": {
                "target_species": len(
                    self._get_image_files(
                        os.path.join(self.test_dir, self.target_class)
                    )
                ),
                "other_species": len(
                    self._get_image_files(os.path.join(self.test_dir, self.other_class))
                ),
            },
        }

        # Calculate totals
        stats["train"]["total"] = (
            stats["train"]["target_species"] + stats["train"]["other_species"]
        )
        stats["test"]["total"] = (
            stats["test"]["target_species"] + stats["test"]["other_species"]
        )

        return stats

    def print_dataset_summary(self):
        """
        Print a summary of the dataset.
        """
        stats = self.get_dataset_statistics()

        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)

        print("\nTraining Set:")
        print(f"  Target Species:  {stats['train']['target_species']:>6} images")
        print(f"  Other Species:   {stats['train']['other_species']:>6} images")
        print(f"  Total:           {stats['train']['total']:>6} images")

        print("\nTest Set:")
        print(f"  Target Species:  {stats['test']['target_species']:>6} images")
        print(f"  Other Species:   {stats['test']['other_species']:>6} images")
        print(f"  Total:           {stats['test']['total']:>6} images")

        print("\n" + "=" * 60)

        return stats

    def create_synthetic_dataset(self, n_samples_per_class=100):
        """
        Create a synthetic dataset for testing purposes.
        This generates random colored images to simulate plant images.

        Args:
            n_samples_per_class (int): Number of samples per class
        """
        from PIL import Image

        print("Creating synthetic dataset for testing...")

        # Create directories
        os.makedirs(os.path.join(self.train_dir, self.target_class), exist_ok=True)
        os.makedirs(os.path.join(self.train_dir, self.other_class), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, self.target_class), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, self.other_class), exist_ok=True)

        # Generate synthetic images
        image_size = (224, 224)

        for split, n_samples in [
            ("train", n_samples_per_class),
            ("test", n_samples_per_class // 5),
        ]:
            split_dir = self.train_dir if split == "train" else self.test_dir

            # Generate target species images (e.g., blueish tones)
            for i in range(n_samples):
                img_array = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
                img_array[:, :, 2] = np.clip(
                    img_array[:, :, 2] + 50, 0, 255
                )  # More blue
                img = Image.fromarray(img_array)
                img.save(
                    os.path.join(split_dir, self.target_class, f"synthetic_{i:04d}.png")
                )

            # Generate other species images (e.g., reddish tones)
            for i in range(n_samples):
                img_array = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
                img_array[:, :, 0] = np.clip(
                    img_array[:, :, 0] + 50, 0, 255
                )  # More red
                img = Image.fromarray(img_array)
                img.save(
                    os.path.join(split_dir, self.other_class, f"synthetic_{i:04d}.png")
                )

        print("Synthetic dataset created successfully!")
        self.print_dataset_summary()
