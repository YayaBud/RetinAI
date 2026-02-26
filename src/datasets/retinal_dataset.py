"""
Base dataset class for retinal disease screening datasets.
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class RetinalDataset(Dataset):
    """
    Base dataset class for retinal images.
    
    Args:
        image_dir: Path to directory containing images
        labels_file: Path to CSV file with image labels
        transform: Optional transform to apply to images
        
    Returns:
        Tuple of (image_tensor, label)
    """
    
    def __init__(self, image_dir, labels_file, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_dir: Directory containing image files
            labels_file: CSV file with columns 'image_id' and 'label'
            transform: Optional torchvision transforms
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Load labels from CSV
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        try:
            self.labels_df = pd.read_csv(labels_file)
        except Exception as e:
            raise ValueError(f"Failed to read labels file: {e}")
        
        # Validate required columns
        if 'image_id' not in self.labels_df.columns or 'label' not in self.labels_df.columns:
            raise ValueError("Labels CSV must contain 'image_id' and 'label' columns")
        
        # Validate dataset is not empty
        if len(self.labels_df) == 0:
            raise ValueError("Dataset contains zero samples")
        
        # Validate label types
        if not pd.api.types.is_integer_dtype(self.labels_df['label']):
            raise TypeError("Labels must be integers")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label)
        """
        # Get image info
        row = self.labels_df.iloc[idx]
        image_id = row['image_id']
        label = int(row['label'])
        
        # Construct image path
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        
        # Try alternative extensions if .jpg doesn't exist
        if not os.path.exists(image_path):
            for ext in ['.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                alt_path = os.path.join(self.image_dir, f"{image_id}{ext}")
                if os.path.exists(alt_path):
                    image_path = alt_path
                    break
        
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(image_path)
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label



class EyePACSDataset(RetinalDataset):
    """
    Dataset for Diabetic Retinopathy detection (5 classes).
    
    Classes:
        0: No DR
        1: Mild
        2: Moderate
        3: Severe
        4: Proliferative DR
    """
    
    def __init__(self, image_dir, labels_file, transform=None):
        super().__init__(image_dir, labels_file, transform)
        
        # Validate labels are in valid range [0, 4]
        invalid_labels = self.labels_df[~self.labels_df['label'].isin([0, 1, 2, 3, 4])]
        if len(invalid_labels) > 0:
            raise ValueError(f"EyePACS labels must be in range [0, 4]. Found invalid labels: {invalid_labels['label'].unique()}")


class REFUGEDataset(RetinalDataset):
    """
    Dataset for Glaucoma detection (binary classification).
    
    Classes:
        0: Normal
        1: Glaucoma
    """
    
    def __init__(self, image_dir, labels_file, transform=None):
        super().__init__(image_dir, labels_file, transform)
        
        # Validate labels are in valid range [0, 1]
        invalid_labels = self.labels_df[~self.labels_df['label'].isin([0, 1])]
        if len(invalid_labels) > 0:
            raise ValueError(f"REFUGE labels must be in range [0, 1]. Found invalid labels: {invalid_labels['label'].unique()}")


class PALMDataset(RetinalDataset):
    """
    Dataset for Pathologic Myopia detection (binary classification).
    
    Classes:
        0: Normal
        1: Pathologic Myopia
    """
    
    def __init__(self, image_dir, labels_file, transform=None):
        super().__init__(image_dir, labels_file, transform)
        
        # Validate labels are in valid range [0, 1]
        invalid_labels = self.labels_df[~self.labels_df['label'].isin([0, 1])]
        if len(invalid_labels) > 0:
            raise ValueError(f"PALM labels must be in range [0, 1]. Found invalid labels: {invalid_labels['label'].unique()}")
