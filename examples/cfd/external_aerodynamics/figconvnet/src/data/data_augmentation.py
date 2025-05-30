import torch
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass

class DataAugmentation:
    """
    Class encapsulating various data augmentation techniques for point clouds.
    """

    @staticmethod
    def translate_pointcloud(pointcloud: torch.Tensor,
                             translation_range: Tuple[float, float] = (2. / 3., 3. / 2.)) -> torch.Tensor:
        """
        Translates the pointcloud by a random factor within a given range.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            translation_range: A tuple specifying the range for translation factors.

        Returns:
            Translated point cloud as a torch.Tensor.
        """
        # Randomly choose translation factors and apply them to the pointcloud
        xyz1 = np.random.uniform(low=translation_range[0], high=translation_range[1], size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return torch.tensor(translated_pointcloud, dtype=torch.float32)

    @staticmethod
    def jitter_pointcloud(pointcloud: torch.Tensor, sigma: float = 0.01, clip: float = 0.02) -> torch.Tensor:
        """
        Adds Gaussian noise to the pointcloud.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            sigma: Standard deviation of the Gaussian noise.
            clip: Maximum absolute value for noise.

        Returns:
            Jittered point cloud as a torch.Tensor.
        """
        # Add Gaussian noise and clip to the specified range
        N, C = pointcloud.shape
        jittered_pointcloud = pointcloud + torch.clamp(sigma * torch.randn(N, C), -clip, clip)
        return jittered_pointcloud

    @staticmethod
    def drop_points(pointcloud: torch.Tensor, drop_rate: float = 0.1) -> torch.Tensor:
        """
        Randomly removes points from the point cloud based on the drop rate.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            drop_rate: The percentage of points to be randomly dropped.

        Returns:
            The point cloud with points dropped as a torch.Tensor.
        """
        # Calculate the number of points to drop
        num_drop = int(drop_rate * pointcloud.size(0))
        # Generate random indices for points to drop
        drop_indices = np.random.choice(pointcloud.size(0), num_drop, replace=False)
        keep_indices = np.setdiff1d(np.arange(pointcloud.size(0)), drop_indices)
        dropped_pointcloud = pointcloud[keep_indices, :]
        return dropped_pointcloud

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    enable_translation: bool = True
    translation_range: Tuple[float, float] = (2./3., 3./2.)
    enable_jitter: bool = True
    jitter_sigma: float = 0.01
    jitter_clip: float = 0.02
    enable_drop: bool = False
    drop_rate: float = 0.1

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'AugmentationConfig':
        """Create an AugmentationConfig instance from a dictionary."""
        return cls(
            enable_translation=config_dict.get('enable_translation', True),
            translation_range=config_dict.get('translation_range', (2./3., 3./2.)),
            enable_jitter=config_dict.get('enable_jitter', True),
            jitter_sigma=config_dict.get('jitter_sigma', 0.01),
            jitter_clip=config_dict.get('jitter_clip', 0.02),
            enable_drop=config_dict.get('enable_drop', False),
            drop_rate=config_dict.get('drop_rate', 0.1)
        )

class DrivAerNetAugmentationPreprocessor:
    """
    Preprocessor for applying data augmentation to point clouds during training.
    """
    def __init__(self, 
                 config: Optional[Union[AugmentationConfig, Dict]] = None,
                 training: bool = True):
        """
        Initialize the preprocessor with augmentation configuration.

        Args:
            config: Configuration for data augmentation, can be either AugmentationConfig instance or dict
            training: Whether in training mode (augmentation only applied during training)
        """
        if config is None:
            self.config = AugmentationConfig()
        elif isinstance(config, dict):
            self.config = AugmentationConfig.from_dict(config)
        elif isinstance(config, AugmentationConfig):
            self.config = config
        else:
            raise TypeError(f"config must be either dict or AugmentationConfig, got {type(config)}")
        
        self.training = training
        self.augmentor = DataAugmentation()

    def __call__(self, data_dict: Dict) -> Dict:
        """
        Apply data augmentation to the point cloud data.

        Args:
            data_dict: Dictionary containing the point cloud data

        Returns:
            Augmented data dictionary
        """
        if not self.training:
            return data_dict

        vertices = data_dict["cell_centers"]

        # Apply augmentations based on config
        if self.config.enable_translation:
            vertices = self.augmentor.translate_pointcloud(
                vertices, self.config.translation_range)
        
        if self.config.enable_jitter:
            vertices = self.augmentor.jitter_pointcloud(
                vertices, self.config.jitter_sigma, self.config.jitter_clip)
        
        if self.config.enable_drop:
            vertices = self.augmentor.drop_points(
                vertices, self.config.drop_rate)
            
        data_dict["cell_centers"] = vertices.numpy()
        return data_dict 