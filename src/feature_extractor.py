"""
Feature Extraction Module for Plant Image Classification

This module provides various feature extraction methods for plant images
to be used with SVM classification.
"""

import cv2
import numpy as np
from PIL import Image
from scipy.stats import kurtosis, skew
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern


class ImageFeatureExtractor:
    """
    Extracts multiple types of features from images for SVM classification.
    """

    def __init__(
        self,
        image_size=(224, 224),
        use_color_histogram=True,
        use_hog=True,
        use_lbp=True,
        use_statistics=True,
    ):
        """
        Initialize the feature extractor.

        Args:
            image_size (tuple): Target size for resizing images (width, height)
            use_color_histogram (bool): Extract color histogram features
            use_hog (bool): Extract HOG (Histogram of Oriented Gradients) features
            use_lbp (bool): Extract LBP (Local Binary Pattern) features
            use_statistics (bool): Extract statistical features
        """
        self.image_size = image_size
        self.use_color_histogram = use_color_histogram
        self.use_hog = use_hog
        self.use_lbp = use_lbp
        self.use_statistics = use_statistics

        # HOG parameters
        self.hog_orientations = 9
        self.hog_pixels_per_cell = (8, 8)
        self.hog_cells_per_block = (2, 2)

        # LBP parameters
        self.lbp_points = 24
        self.lbp_radius = 3

        # Color histogram parameters
        self.hist_bins = 32

    def extract_features(self, image_path):
        """
        Extract all enabled features from an image.

        Args:
            image_path (str): Path to the image file

        Returns:
            numpy.ndarray: Concatenated feature vector
        """
        # Load and preprocess image
        image = self._load_and_preprocess(image_path)

        features = []

        # Extract color histogram features
        if self.use_color_histogram:
            color_features = self._extract_color_histogram(image)
            features.append(color_features)

        # Extract HOG features
        if self.use_hog:
            hog_features = self._extract_hog(image)
            features.append(hog_features)

        # Extract LBP features
        if self.use_lbp:
            lbp_features = self._extract_lbp(image)
            features.append(lbp_features)

        # Extract statistical features
        if self.use_statistics:
            stat_features = self._extract_statistics(image)
            features.append(stat_features)

        # Concatenate all features
        feature_vector = np.concatenate(features)

        return feature_vector

    def _load_and_preprocess(self, image_path):
        """
        Load and preprocess an image.

        Args:
            image_path (str): Path to the image file

        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Load image using PIL and convert to RGB
        image = Image.open(image_path).convert("RGB")

        # Resize to target size
        image = image.resize(self.image_size, Image.LANCZOS)

        # Convert to numpy array
        image = np.array(image)

        return image

    def _extract_color_histogram(self, image):
        """
        Extract color histogram features from RGB channels.

        Args:
            image (numpy.ndarray): RGB image

        Returns:
            numpy.ndarray: Color histogram features
        """
        histograms = []

        # Extract histogram for each channel (R, G, B)
        for channel in range(3):
            hist, _ = np.histogram(
                image[:, :, channel], bins=self.hist_bins, range=(0, 256)
            )
            # Normalize histogram
            hist = hist.astype(float) / hist.sum()
            histograms.append(hist)

        # Concatenate all channel histograms
        color_features = np.concatenate(histograms)

        return color_features

    def _extract_hog(self, image):
        """
        Extract Histogram of Oriented Gradients (HOG) features.

        Args:
            image (numpy.ndarray): RGB image

        Returns:
            numpy.ndarray: HOG features
        """
        # Convert to grayscale
        gray = rgb2gray(image)

        # Extract HOG features
        hog_features = hog(
            gray,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            block_norm="L2-Hys",
            visualize=False,
            feature_vector=True,
        )

        return hog_features

    def _extract_lbp(self, image):
        """
        Extract Local Binary Pattern (LBP) features.

        Args:
            image (numpy.ndarray): RGB image

        Returns:
            numpy.ndarray: LBP histogram features
        """
        # Convert to grayscale
        gray = rgb2gray(image)

        # Convert to uint8
        gray_uint8 = (gray * 255).astype(np.uint8)

        # Compute LBP
        lbp = local_binary_pattern(
            gray_uint8, self.lbp_points, self.lbp_radius, method="uniform"
        )

        # Compute LBP histogram
        n_bins = self.lbp_points + 2  # uniform patterns + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

        # Normalize histogram
        hist = hist.astype(float) / hist.sum()

        return hist

    def _extract_statistics(self, image):
        """
        Extract statistical features from the image.

        Args:
            image (numpy.ndarray): RGB image

        Returns:
            numpy.ndarray: Statistical features
        """
        features = []

        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Extract statistics for each channel in each color space
        for color_space, name in [(image, "RGB"), (hsv, "HSV"), (lab, "LAB")]:
            for channel in range(3):
                channel_data = color_space[:, :, channel].flatten()

                # Mean, std, min, max
                features.extend(
                    [
                        np.mean(channel_data),
                        np.std(channel_data),
                        np.min(channel_data),
                        np.max(channel_data),
                        skew(channel_data),
                        kurtosis(channel_data),
                    ]
                )

        return np.array(features)

    def get_feature_dimension(self):
        """
        Calculate the total dimension of the feature vector.

        Returns:
            int: Total number of features
        """
        # Create a dummy image to extract features and get dimension
        dummy_image = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)

        features = []

        if self.use_color_histogram:
            color_features = self._extract_color_histogram(dummy_image)
            features.append(len(color_features))

        if self.use_hog:
            hog_features = self._extract_hog(dummy_image)
            features.append(len(hog_features))

        if self.use_lbp:
            lbp_features = self._extract_lbp(dummy_image)
            features.append(len(lbp_features))

        if self.use_statistics:
            stat_features = self._extract_statistics(dummy_image)
            features.append(len(stat_features))

        return sum(features)


def extract_features_from_directory(directory_path, feature_extractor):
    """
    Extract features from all images in a directory.

    Args:
        directory_path (str): Path to directory containing images
        feature_extractor (ImageFeatureExtractor): Feature extractor instance

    Returns:
        tuple: (features array, file paths list)
    """
    import os

    from tqdm import tqdm

    # Get all image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    image_files = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))

    # Extract features from all images
    features_list = []
    valid_files = []

    print(f"Extracting features from {len(image_files)} images...")

    for image_path in tqdm(image_files):
        try:
            features = feature_extractor.extract_features(image_path)
            features_list.append(features)
            valid_files.append(image_path)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    # Convert to numpy array
    features_array = np.array(features_list)

    return features_array, valid_files
