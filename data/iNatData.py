from collections import Counter
import os
import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets import VisionDataset


class INaturalistNClasses(VisionDataset):
    """
    Adaptation of the PyTorch INaturalist dataset implementation that contains only
    data from a specific taxonomy subtree (new classes).

    Args:
        root (str): Root directory path containing the dataset
        split (str, optional): The dataset split, either "train" or "val". Default: "train"
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. Default: None
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it. Default: None
        classes (List[str], optional): List of class names to include. If None, uses all classes.
            Default: None

    Note:
        This dataset uses the validation set as test set according to the paper 
        "When Does Contrastive Visual Representation Learning Work?"
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        classes: Optional[List[str]] = None,
    ) -> None:
        self.split = split
        self.classes = classes
        
        if self.split not in ["train", "val"]:
            raise ValueError(f"Split must be 'train' or 'val', got {split}")

        # Robust split directory resolution
        split_path = self._resolve_split_path(root, split)
        super().__init__(split_path, transform=transform, target_transform=target_transform)

        if not self._check_integrity():
            raise RuntimeError(f"Dataset directory empty or invalid: {self.root}")

        # List all valid category subdirectories (sorted for deterministic behavior)
        self.all_categories: List[str] = sorted([
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ])

        if len(self.all_categories) == 0:
            raise RuntimeError(f"No category subdirectories found in: {self.root}")

        # Index of all samples: List of (class_id, cat_idx, filename)
        # where cat_idx is the index into self.all_categories
        self.index: List[Tuple[int, int, str]] = []

        if classes is None:
            # Use full dataset
            print("Using full dataset!")
            self.classes = [
                cat.split("_", 1)[1].lower() if "_" in cat else cat.lower()
                for cat in self.all_categories
            ]
            for cat_idx in range(len(self.all_categories)):
                self._add_category_to_index(cat_idx, cat_idx)
        else:
            # Only add samples from specified classes
            for cls_id, cls in enumerate(classes):
                categories_for_cls = self._get_categories_for_class(cls)
                for cat_idx in categories_for_cls:
                    self._add_category_to_index(cls_id, cat_idx)

        self._print_dataset_info()

    def _resolve_split_path(self, root: str, split: str) -> str:
        """Find the actual split directory supporting multiple directory layouts."""
        candidates = []
        if split == "train":
            candidates = [
                os.path.join(root, "train_mini"),
                os.path.join(root, "train"),
                os.path.join(root, "inat21", "train_mini"),
                os.path.join(root, "inat21", "train"),
                os.path.join("data", "inat21", "train_mini"),
                os.path.join("data", "inat21", "train"),
                os.path.join(root, split),
            ]
        elif split == "val":
            candidates = [
                os.path.join(root, "val"),
                os.path.join(root, "val", "val"),
                os.path.join(root, "val_mini"),
                os.path.join(root, "inat21", "val"),
                os.path.join(root, "inat21", "val", "val"),
                os.path.join("data", "inat21", "val"),
                os.path.join("data", "inat21", "val", "val"),
                os.path.join(root, split),
            ]

        # Check if root itself is already the split directory
        norm_root = os.path.normpath(root)
        if os.path.basename(norm_root) in [split, "train_mini", "val"]:
            if os.path.isdir(root) and any(os.path.isdir(os.path.join(root, d)) for d in os.listdir(root)):
                return root

        for cand in candidates:
            if os.path.isdir(cand):
                subdirs = [d for d in os.listdir(cand) if os.path.isdir(os.path.join(cand, d))]
                if len(subdirs) > 0:
                    return cand

        # If none found, provide detailed diagnostic error
        searched = "\n  - ".join([os.path.abspath(c) for c in candidates])
        raise FileNotFoundError(
            f"Could not find valid '{split}' directory for iNaturalist dataset in '{root}'.\n"
            f"Searched following candidate locations:\n  - {searched}\n"
            f"Please check if data is downloaded and extracted properly."
        )

    def _add_category_to_index(self, cls_id: int, cat_idx: int) -> None:
        """Add all images from a category to the index."""
        cat_dir_name = self.all_categories[cat_idx]
        cat_path = os.path.join(self.root, cat_dir_name)
        try:
            files = sorted(os.listdir(cat_path))
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.index.append((cls_id, cat_idx, fname))
        except FileNotFoundError:
            print(f"Warning: Category directory not found: {cat_path}")

    def _print_dataset_info(self) -> None:
        """Print dataset information and class statistics."""
        print(f'Created dataset {self.__class__.__name__} [{self.split}] with {len(self)} samples from {self.root}.')
        cls_counter = Counter(cls_id for (cls_id, _, _) in self.index)
        if self.classes:
            cls_counts = [cls_counter.get(i, 0) for i in range(len(self.classes))]
            print(f'Classes: {self.classes}')
            print(f'Class counts: {cls_counts}')

    def _get_categories_for_class(self, cls: str) -> List[int]:
        """
        Returns list of category indices in self.all_categories that match the class name.
        
        Args:
            cls (str): The class name to search for (e.g. 'Plantae_Tracheophyta_...')
            
        Returns:
            List[int]: Indices in self.all_categories
        """
        matched_indices: List[int] = []
        cls_lower = cls.lower()
        
        for dir_idx, dir_name in enumerate(self.all_categories):
            # Remove the numeric prefix (e.g. "07018_Plantae_..." -> "Plantae_...")
            if "_" in dir_name:
                cat_name = dir_name.split("_", 1)[1].lower()
            else:
                cat_name = dir_name.lower()

            if cat_name.startswith(cls_lower) or cls_lower in cat_name:
                matched_indices.append(dir_idx)

        return matched_indices

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get item at the specified index.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, target) where target is the class index
        """
        class_id, cat_idx, fname = self.index[index]
        cat_dir_name = self.all_categories[cat_idx]
        img_path = os.path.join(self.root, cat_dir_name, fname)
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        target = class_id

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.index)

    def _check_integrity(self) -> bool:
        """Check if the dataset directory exists and is not empty."""
        return os.path.exists(self.root) and len(os.listdir(self.root)) > 0
