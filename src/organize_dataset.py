"""
Dataset Organization Utility Script

This script helps organize plant images into the proper structure for training
the SVM classifier. It handles splitting data into train/test sets and organizing
images into target_species and other_species directories.
"""

import argparse
import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


def get_image_files(directory, extensions=None):
    """
    Get all image files from a directory.

    Args:
        directory (str): Path to directory
        extensions (list): List of file extensions to include

    Returns:
        list: List of image file paths
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

    image_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))

    return sorted(image_files)


def organize_single_class(
    source_dir,
    data_root,
    class_label,
    test_size=0.2,
    random_state=42,
    copy=True,
    verbose=True,
):
    """
    Organize images from a single source directory into train/test splits.

    Args:
        source_dir (str): Source directory containing images
        data_root (str): Root directory for organized data
        class_label (str): 'target_species' or 'other_species'
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        copy (bool): If True, copy files; if False, move files
        verbose (bool): Print progress information

    Returns:
        dict: Statistics about organized data
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if class_label not in ["target_species", "other_species"]:
        raise ValueError("class_label must be 'target_species' or 'other_species'")

    # Get all image files
    image_files = get_image_files(source_dir)

    if len(image_files) == 0:
        print(f"No images found in {source_dir}")
        return None

    if verbose:
        print(f"\nFound {len(image_files)} images in {source_dir}")

    # Split into train and test
    train_files, test_files = train_test_split(
        image_files, test_size=test_size, random_state=random_state
    )

    # Create destination directories
    train_dir = os.path.join(data_root, "train", class_label)
    test_dir = os.path.join(data_root, "test", class_label)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy/move files
    operation = "Copying" if copy else "Moving"

    if verbose:
        print(f"{operation} {len(train_files)} images to training set...")

    for file_path in train_files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(train_dir, filename)

        # Handle duplicate filenames
        counter = 1
        base_name, ext = os.path.splitext(filename)
        while os.path.exists(dest_path):
            new_filename = f"{base_name}_{counter}{ext}"
            dest_path = os.path.join(train_dir, new_filename)
            counter += 1

        if copy:
            shutil.copy2(file_path, dest_path)
        else:
            shutil.move(file_path, dest_path)

    if verbose:
        print(f"{operation} {len(test_files)} images to test set...")

    for file_path in test_files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(test_dir, filename)

        # Handle duplicate filenames
        counter = 1
        base_name, ext = os.path.splitext(filename)
        while os.path.exists(dest_path):
            new_filename = f"{base_name}_{counter}{ext}"
            dest_path = os.path.join(test_dir, new_filename)
            counter += 1

        if copy:
            shutil.copy2(file_path, dest_path)
        else:
            shutil.move(file_path, dest_path)

    stats = {
        "source_dir": source_dir,
        "class_label": class_label,
        "total_images": len(image_files),
        "train_images": len(train_files),
        "test_images": len(test_files),
        "operation": "copy" if copy else "move",
    }

    if verbose:
        print(f"✓ Organization complete for {class_label}")
        print(f"  Training: {len(train_files)} images")
        print(f"  Test: {len(test_files)} images")

    return stats


def organize_multiple_classes(
    source_dirs,
    data_root,
    class_labels,
    test_size=0.2,
    random_state=42,
    copy=True,
    verbose=True,
):
    """
    Organize images from multiple source directories.

    Args:
        source_dirs (list): List of source directories
        data_root (str): Root directory for organized data
        class_labels (list): List of class labels corresponding to source_dirs
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        copy (bool): If True, copy files; if False, move files
        verbose (bool): Print progress information

    Returns:
        list: List of statistics dictionaries
    """
    if len(source_dirs) != len(class_labels):
        raise ValueError(
            "Number of source directories must match number of class labels"
        )

    all_stats = []

    for source_dir, class_label in zip(source_dirs, class_labels):
        stats = organize_single_class(
            source_dir=source_dir,
            data_root=data_root,
            class_label=class_label,
            test_size=test_size,
            random_state=random_state,
            copy=copy,
            verbose=verbose,
        )

        if stats:
            all_stats.append(stats)

    return all_stats


def print_dataset_statistics(data_root):
    """
    Print statistics about the organized dataset.

    Args:
        data_root (str): Root directory of organized data
    """
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    # Count images
    train_target = len(get_image_files(os.path.join(train_dir, "target_species")))
    train_other = len(get_image_files(os.path.join(train_dir, "other_species")))
    test_target = len(get_image_files(os.path.join(test_dir, "target_species")))
    test_other = len(get_image_files(os.path.join(test_dir, "other_species")))

    total_train = train_target + train_other
    total_test = test_target + test_other
    total = total_train + total_test

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print("\nTraining Set:")
    print(f"  Target Species:  {train_target:>6} images")
    print(f"  Other Species:   {train_other:>6} images")
    print(f"  Total:           {total_train:>6} images")

    print("\nTest Set:")
    print(f"  Target Species:  {test_target:>6} images")
    print(f"  Other Species:   {test_other:>6} images")
    print(f"  Total:           {total_test:>6} images")

    print("\nOverall:")
    print(f"  Total Images:    {total:>6}")
    if total > 0:
        print(
            f"  Train/Test Split: {total_train}/{total_test} ({total_train / total * 100:.1f}%/{total_test / total * 100:.1f}%)"
        )
        print(
            f"  Target/Other:     {train_target + test_target}/{train_other + test_other} ({(train_target + test_target) / total * 100:.1f}%/{(train_other + test_other) / total * 100:.1f}%)"
        )

    print("=" * 60 + "\n")


def interactive_mode(data_root):
    """
    Interactive mode for organizing datasets.

    Args:
        data_root (str): Root directory for organized data
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE DATASET ORGANIZATION")
    print("=" * 60)

    print("\nThis utility will help you organize your plant images.")
    print("Images will be split into training and test sets.")

    # Get source directory
    while True:
        source_dir = input("\nEnter the path to your source image directory: ").strip()
        if os.path.exists(source_dir):
            image_count = len(get_image_files(source_dir))
            if image_count > 0:
                print(f"✓ Found {image_count} images in {source_dir}")
                break
            else:
                print("✗ No images found in that directory. Please try again.")
        else:
            print("✗ Directory not found. Please try again.")

    # Get class label
    while True:
        print("\nWhat type of species are these images?")
        print("  1. Target species (the species you want to identify)")
        print("  2. Other species (background/negative class)")
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            class_label = "target_species"
            break
        elif choice == "2":
            class_label = "other_species"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    # Get test size
    while True:
        test_size_input = input(
            "\nEnter test set percentage (e.g., 20 for 20%): "
        ).strip()
        try:
            test_size = float(test_size_input) / 100
            if 0 < test_size < 1:
                break
            else:
                print("Please enter a value between 0 and 100.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Copy or move
    while True:
        print("\nDo you want to copy or move the files?")
        print("  1. Copy (keeps original files)")
        print("  2. Move (removes files from source)")
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            copy = True
            break
        elif choice == "2":
            copy = False
            print("\n⚠️  WARNING: This will MOVE files from the source directory!")
            confirm = input("Are you sure? (yes/no): ").strip().lower()
            if confirm == "yes":
                break
            else:
                print("Operation cancelled. Using copy instead.")
                copy = True
                break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    # Confirm and execute
    print("\n" + "-" * 60)
    print("SUMMARY:")
    print(f"  Source: {source_dir}")
    print(f"  Class: {class_label}")
    print(f"  Test size: {test_size * 100:.0f}%")
    print(f"  Operation: {'Copy' if copy else 'Move'}")
    print(f"  Destination: {data_root}")
    print("-" * 60)

    confirm = input("\nProceed with organization? (yes/no): ").strip().lower()

    if confirm == "yes":
        stats = organize_single_class(
            source_dir=source_dir,
            data_root=data_root,
            class_label=class_label,
            test_size=test_size,
            random_state=42,
            copy=copy,
            verbose=True,
        )

        print("\n✓ Dataset organization complete!")
        print_dataset_statistics(data_root)

        # Ask if they want to add more
        more = input("Do you want to add more images? (yes/no): ").strip().lower()
        if more == "yes":
            interactive_mode(data_root)
    else:
        print("\nOperation cancelled.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Organize plant images for SVM classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python organize_dataset.py --interactive

  # Organize target species images
  python organize_dataset.py --source src/Myosotis_sylvatica --class target_species

  # Organize with custom test size
  python organize_dataset.py --source images/ --class other_species --test-size 0.3

  # Move files instead of copying
  python organize_dataset.py --source images/ --class target_species --move

  # Show current dataset statistics
  python organize_dataset.py --stats
        """,
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="../data",
        help="Root directory for organized data (default: ../data)",
    )

    parser.add_argument("--source", type=str, help="Source directory containing images")

    parser.add_argument(
        "--class",
        dest="class_label",
        type=str,
        choices=["target_species", "other_species"],
        help="Class label for the images",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for test set (default: 0.2)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--move", action="store_true", help="Move files instead of copying"
    )

    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )

    parser.add_argument(
        "--stats", action="store_true", help="Show dataset statistics only"
    )

    args = parser.parse_args()

    # Create data root if it doesn't exist
    os.makedirs(args.data_root, exist_ok=True)

    # Show statistics only
    if args.stats:
        print_dataset_statistics(args.data_root)
        return

    # Interactive mode
    if args.interactive:
        interactive_mode(args.data_root)
        return

    # Validate arguments for non-interactive mode
    if not args.source or not args.class_label:
        parser.error("--source and --class are required (or use --interactive mode)")

    # Organize dataset
    print("\n" + "=" * 60)
    print("ORGANIZING DATASET")
    print("=" * 60)

    stats = organize_single_class(
        source_dir=args.source,
        data_root=args.data_root,
        class_label=args.class_label,
        test_size=args.test_size,
        random_state=args.random_state,
        copy=not args.move,
        verbose=True,
    )

    if stats:
        print("\n✓ Dataset organization complete!")
        print_dataset_statistics(args.data_root)


if __name__ == "__main__":
    main()
