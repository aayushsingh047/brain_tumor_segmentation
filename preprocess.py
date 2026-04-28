"""
MRI Preprocessing Pipeline
Handles N4 bias correction, normalization, and resampling
"""

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage
from skimage import exposure
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRIPreprocessor:
    """
    Complete preprocessing pipeline for brain MRI scans
    """
    
    def __init__(self, target_spacing=(1.0, 1.0, 1.0), target_shape=(128, 128, 128)):
        """
        Initialize preprocessor
        
        Args:
            target_spacing: desired voxel spacing in mm
            target_shape: desired output shape
        """
        self.target_spacing = target_spacing
        self.target_shape = target_shape
        
    def load_nifti(self, filepath):
        """
        Load NIfTI file
        
        Args:
            filepath: path to .nii or .nii.gz file
            
        Returns:
            data: numpy array
            affine: affine transformation matrix
            header: NIfTI header
        """
        try:
            img = nib.load(filepath)
            data = img.get_fdata()
            affine = img.affine
            header = img.header
            
            logger.info(f"Loaded MRI with shape: {data.shape}")
            return data, affine, header
        except Exception as e:
            logger.error(f"Error loading NIfTI file: {e}")
            raise
    
    def n4_bias_correction(self, data):
        """
        Apply N4 bias field correction using SimpleITK
        
        Args:
            data: numpy array
            
        Returns:
            corrected data
        """
        try:
            # Convert to SimpleITK image
            sitk_img = sitk.GetImageFromArray(data.astype(np.float32))
            
            # Create mask (non-zero regions)
            mask = sitk.OtsuThreshold(sitk_img, 0, 1)
            
            # Apply N4 bias field correction
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
            
            corrected_img = corrector.Execute(sitk_img, mask)
            
            # Convert back to numpy
            corrected_data = sitk.GetArrayFromImage(corrected_img)
            
            logger.info("N4 bias correction completed")
            return corrected_data
        except Exception as e:
            logger.warning(f"N4 correction failed: {e}. Returning original data.")
            return data
    
    def skull_stripping(self, data, threshold_percentile=5):
        """
        Simple skull stripping using intensity thresholding
        
        Args:
            data: numpy array
            threshold_percentile: percentile for thresholding
            
        Returns:
            stripped data, brain mask
        """
        # Calculate threshold
        threshold = np.percentile(data[data > 0], threshold_percentile)
        
        # Create initial mask
        mask = data > threshold
        
        # Morphological operations
        struct = ndimage.generate_binary_structure(3, 2)
        mask = ndimage.binary_erosion(mask, structure=struct, iterations=2)
        mask = ndimage.binary_dilation(mask, structure=struct, iterations=3)
        
        # Keep largest connected component
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            largest_component = np.argmax(sizes) + 1
            mask = labeled == largest_component
        
        # Apply mask
        stripped_data = data * mask
        
        logger.info("Skull stripping completed")
        return stripped_data, mask
    
    def normalize_intensity(self, data, method='zscore'):
        """
        Normalize intensity values
        
        Args:
            data: numpy array
            method: 'zscore', 'minmax', or 'percentile'
            
        Returns:
            normalized data
        """
        # Work only with non-zero voxels
        nonzero_mask = data > 0
        
        if method == 'zscore':
            mean = np.mean(data[nonzero_mask])
            std = np.std(data[nonzero_mask])
            normalized = np.zeros_like(data, dtype=np.float32)
            normalized[nonzero_mask] = (data[nonzero_mask] - mean) / (std + 1e-8)
            
        elif method == 'minmax':
            min_val = np.min(data[nonzero_mask])
            max_val = np.max(data[nonzero_mask])
            normalized = np.zeros_like(data, dtype=np.float32)
            normalized[nonzero_mask] = (data[nonzero_mask] - min_val) / (max_val - min_val + 1e-8)
            
        elif method == 'percentile':
            p1 = np.percentile(data[nonzero_mask], 1)
            p99 = np.percentile(data[nonzero_mask], 99)
            normalized = np.clip(data, p1, p99)
            normalized = (normalized - p1) / (p99 - p1 + 1e-8)
            
        else:
            normalized = data
        
        logger.info(f"Intensity normalization ({method}) completed")
        return normalized
    
    def resample(self, data, original_spacing=(1.0, 1.0, 1.0)):
        """
        Resample to target spacing and shape
        
        Args:
            data: numpy array
            original_spacing: original voxel spacing
            
        Returns:
            resampled data
        """
        # Calculate zoom factors
        zoom_factors = [
            o / t for o, t in zip(original_spacing, self.target_spacing)
        ]
        
        # Resample
        resampled = ndimage.zoom(data, zoom_factors, order=1)
        
        # Resize to target shape
        current_shape = resampled.shape
        
        if current_shape != self.target_shape:
            # Calculate padding/cropping
            pad_width = []
            for curr, target in zip(current_shape, self.target_shape):
                if curr < target:
                    pad_before = (target - curr) // 2
                    pad_after = target - curr - pad_before
                    pad_width.append((pad_before, pad_after))
                else:
                    pad_width.append((0, 0))
            
            # Pad if necessary
            if any(p[0] > 0 or p[1] > 0 for p in pad_width):
                resampled = np.pad(resampled, pad_width, mode='constant', constant_values=0)
            
            # Crop if necessary
            slices = []
            for curr, target in zip(resampled.shape, self.target_shape):
                if curr > target:
                    start = (curr - target) // 2
                    slices.append(slice(start, start + target))
                else:
                    slices.append(slice(None))
            
            resampled = resampled[tuple(slices)]
        
        logger.info(f"Resampled to shape: {resampled.shape}")
        return resampled
    
    def enhance_contrast(self, data):
        """
        Enhance contrast using adaptive histogram equalization
        
        Args:
            data: numpy array
            
        Returns:
            contrast-enhanced data
        """
        # Work with normalized data
        nonzero_mask = data > 0
        
        if np.sum(nonzero_mask) == 0:
            return data
        
        # Rescale to 0-1 for CLAHE
        min_val = np.min(data[nonzero_mask])
        max_val = np.max(data[nonzero_mask])
        
        if max_val - min_val < 1e-6:
            return data
        
        data_scaled = np.zeros_like(data)
        data_scaled[nonzero_mask] = (data[nonzero_mask] - min_val) / (max_val - min_val)
        
        # Apply CLAHE slice by slice
        enhanced = np.zeros_like(data_scaled)
        for i in range(data.shape[2]):
            if np.sum(nonzero_mask[:, :, i]) > 0:
                enhanced[:, :, i] = exposure.equalize_adapthist(
                    data_scaled[:, :, i], 
                    clip_limit=0.03
                )
        
        # Scale back
        enhanced[nonzero_mask] = enhanced[nonzero_mask] * (max_val - min_val) + min_val
        
        logger.info("Contrast enhancement completed")
        return enhanced
    
    def preprocess(self, filepath, apply_n4=True, apply_skull_strip=True, 
                   enhance=False, normalize_method='zscore'):
        """
        Run complete preprocessing pipeline
        
        Args:
            filepath: path to NIfTI file
            apply_n4: whether to apply N4 bias correction
            apply_skull_strip: whether to apply skull stripping
            enhance: whether to enhance contrast
            normalize_method: normalization method
            
        Returns:
            processed_data: preprocessed numpy array
            metadata: dict with preprocessing info
        """
        # Load data
        data, affine, header = self.load_nifti(filepath)
        original_shape = data.shape
        
        # Get voxel spacing from header
        try:
            spacing = header.get_zooms()[:3]
        except:
            spacing = (1.0, 1.0, 1.0)
        
        metadata = {
            'original_shape': original_shape,
            'original_spacing': spacing,
            'steps_applied': []
        }
        
        # N4 bias correction
        if apply_n4:
            data = self.n4_bias_correction(data)
            metadata['steps_applied'].append('n4_correction')
        
        # Skull stripping
        brain_mask = None
        if apply_skull_strip:
            data, brain_mask = self.skull_stripping(data)
            metadata['steps_applied'].append('skull_stripping')
        
        # Contrast enhancement
        if enhance:
            data = self.enhance_contrast(data)
            metadata['steps_applied'].append('contrast_enhancement')
        
        # Normalize
        data = self.normalize_intensity(data, method=normalize_method)
        metadata['steps_applied'].append(f'normalization_{normalize_method}')
        
        # Resample
        data = self.resample(data, original_spacing=spacing)
        metadata['steps_applied'].append('resampling')
        metadata['final_shape'] = data.shape
        metadata['target_spacing'] = self.target_spacing
        
        logger.info(f"Preprocessing complete. Applied steps: {metadata['steps_applied']}")
        
        return data, metadata


def quick_preprocess(filepath, target_shape=(128, 128, 128)):
    """
    Quick preprocessing with default parameters
    
    Args:
        filepath: path to NIfTI file
        target_shape: desired output shape
        
    Returns:
        processed data, metadata
    """
    preprocessor = MRIPreprocessor(target_shape=target_shape)
    return preprocessor.preprocess(
        filepath,
        apply_n4=True,
        apply_skull_strip=True,
        enhance=False,
        normalize_method='zscore'
    )


def create_sample_mri(shape=(128, 128, 128), add_tumor=True):
    """
    Create a synthetic MRI scan for testing
    
    Args:
        shape: output shape
        add_tumor: whether to add synthetic tumor
        
    Returns:
        synthetic MRI data
    """
    # Create base brain structure
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, shape[0]),
        np.linspace(-1, 1, shape[1]),
        np.linspace(-1, 1, shape[2]),
        indexing='ij'
    )
    
    # Brain ellipsoid
    brain = np.exp(-2 * (x**2 + y**2 + 1.5*z**2))
    
    # Add texture
    noise = np.random.randn(*shape) * 0.1
    brain = brain + noise
    
    # Add tumor if requested
    if add_tumor:
        # Tumor location
        tumor_center = (shape[0]//2 + 10, shape[1]//2 - 5, shape[2]//2)
        tumor_radius = 15
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    dist = np.sqrt(
                        (i - tumor_center[0])**2 + 
                        (j - tumor_center[1])**2 + 
                        (k - tumor_center[2])**2
                    )
                    if dist < tumor_radius:
                        intensity = 1.5 * (1 - dist / tumor_radius)
                        brain[i, j, k] += intensity
    
    # Normalize
    brain = (brain - brain.min()) / (brain.max() - brain.min())
    brain = brain * 1000  # Scale to typical MRI range
    
    logger.info(f"Created synthetic MRI with shape: {shape}")
    return brain