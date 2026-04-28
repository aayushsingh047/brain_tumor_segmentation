"""
Utility Functions and Metrics Calculation
"""

import numpy as np
from scipy import ndimage
import nibabel as nib
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_dice_coefficient(pred, target, class_id=None):
    """
    Calculate Dice coefficient
    
    Args:
        pred: predicted segmentation
        target: ground truth segmentation
        class_id: specific class to compute (if None, compute for all non-background)
        
    Returns:
        Dice coefficient
    """
    if class_id is not None:
        pred_binary = (pred == class_id).astype(np.float32)
        target_binary = (target == class_id).astype(np.float32)
    else:
        pred_binary = (pred > 0).astype(np.float32)
        target_binary = (target > 0).astype(np.float32)
    
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary)
    
    if union == 0:
        return 1.0 if np.sum(pred_binary) == 0 else 0.0
    
    dice = (2.0 * intersection) / union
    return dice


def calculate_iou(pred, target, class_id=None):
    """
    Calculate Intersection over Union (IoU)
    
    Args:
        pred: predicted segmentation
        target: ground truth segmentation
        class_id: specific class to compute
        
    Returns:
        IoU score
    """
    if class_id is not None:
        pred_binary = (pred == class_id).astype(np.float32)
        target_binary = (target == class_id).astype(np.float32)
    else:
        pred_binary = (pred > 0).astype(np.float32)
        target_binary = (target > 0).astype(np.float32)
    
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary) - intersection
    
    if union == 0:
        return 1.0 if np.sum(pred_binary) == 0 else 0.0
    
    iou = intersection / union
    return iou


def calculate_sensitivity(pred, target, class_id=None):
    """
    Calculate sensitivity (recall/true positive rate)
    
    Args:
        pred: predicted segmentation
        target: ground truth segmentation
        class_id: specific class
        
    Returns:
        sensitivity
    """
    if class_id is not None:
        pred_binary = (pred == class_id).astype(np.float32)
        target_binary = (target == class_id).astype(np.float32)
    else:
        pred_binary = (pred > 0).astype(np.float32)
        target_binary = (target > 0).astype(np.float32)
    
    true_positives = np.sum(pred_binary * target_binary)
    actual_positives = np.sum(target_binary)
    
    if actual_positives == 0:
        return 1.0 if np.sum(pred_binary) == 0 else 0.0
    
    sensitivity = true_positives / actual_positives
    return sensitivity


def calculate_specificity(pred, target, class_id=None):
    """
    Calculate specificity (true negative rate)
    
    Args:
        pred: predicted segmentation
        target: ground truth segmentation
        class_id: specific class
        
    Returns:
        specificity
    """
    if class_id is not None:
        pred_binary = (pred == class_id).astype(np.float32)
        target_binary = (target == class_id).astype(np.float32)
    else:
        pred_binary = (pred > 0).astype(np.float32)
        target_binary = (target > 0).astype(np.float32)
    
    true_negatives = np.sum((1 - pred_binary) * (1 - target_binary))
    actual_negatives = np.sum(1 - target_binary)
    
    if actual_negatives == 0:
        return 0.0
    
    specificity = true_negatives / actual_negatives
    return specificity


def calculate_volume(segmentation, voxel_spacing=(1.0, 1.0, 1.0), class_id=None):
    """
    Calculate volume of segmented region
    
    Args:
        segmentation: segmentation mask
        voxel_spacing: spacing in mm
        class_id: specific class to compute volume for
        
    Returns:
        volume in mm³ and cm³
    """
    if class_id is not None:
        mask = segmentation == class_id
    else:
        mask = segmentation > 0
    
    num_voxels = np.sum(mask)
    voxel_volume = np.prod(voxel_spacing)  # mm³
    
    volume_mm3 = num_voxels * voxel_volume
    volume_cm3 = volume_mm3 / 1000.0
    
    return volume_mm3, volume_cm3


def calculate_surface_area(segmentation, voxel_spacing=(1.0, 1.0, 1.0), class_id=None):
    """
    Calculate surface area of segmented region
    
    Args:
        segmentation: segmentation mask
        voxel_spacing: spacing in mm
        class_id: specific class
        
    Returns:
        surface area in mm²
    """
    if class_id is not None:
        mask = segmentation == class_id
    else:
        mask = segmentation > 0
    
    # Use gradient to find boundaries
    grad_x = np.abs(np.diff(mask.astype(np.float32), axis=0))
    grad_y = np.abs(np.diff(mask.astype(np.float32), axis=1))
    grad_z = np.abs(np.diff(mask.astype(np.float32), axis=2))
    
    # Calculate surface voxels
    surface_x = np.sum(grad_x) * voxel_spacing[1] * voxel_spacing[2]
    surface_y = np.sum(grad_y) * voxel_spacing[0] * voxel_spacing[2]
    surface_z = np.sum(grad_z) * voxel_spacing[0] * voxel_spacing[1]
    
    surface_area = surface_x + surface_y + surface_z
    
    return surface_area


def calculate_all_metrics(pred, target=None, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Calculate all metrics for segmentation
    
    Args:
        pred: predicted segmentation
        target: ground truth (optional)
        voxel_spacing: voxel spacing in mm
        
    Returns:
        dictionary of metrics
    """
    metrics = {}
    
    # Volume metrics (always available)
    for class_id in range(1, 4):
        volume_mm3, volume_cm3 = calculate_volume(pred, voxel_spacing, class_id)
        metrics[f'volume_class_{class_id}_mm3'] = volume_mm3
        metrics[f'volume_class_{class_id}_cm3'] = volume_cm3
    
    # Total tumor volume
    total_volume_mm3, total_volume_cm3 = calculate_volume(pred, voxel_spacing)
    metrics['total_tumor_volume_mm3'] = total_volume_mm3
    metrics['total_tumor_volume_cm3'] = total_volume_cm3
    
    # Surface area
    metrics['surface_area_mm2'] = calculate_surface_area(pred, voxel_spacing)
    
    # Comparison metrics (if ground truth available)
    if target is not None:
        # Overall metrics
        metrics['dice_overall'] = calculate_dice_coefficient(pred, target)
        metrics['iou_overall'] = calculate_iou(pred, target)
        metrics['sensitivity_overall'] = calculate_sensitivity(pred, target)
        metrics['specificity_overall'] = calculate_specificity(pred, target)
        
        # Per-class metrics
        for class_id in range(1, 4):
            metrics[f'dice_class_{class_id}'] = calculate_dice_coefficient(pred, target, class_id)
            metrics[f'iou_class_{class_id}'] = calculate_iou(pred, target, class_id)
    
    return metrics


def get_tumor_center(segmentation, class_id=None):
    """
    Get center of mass of tumor
    
    Args:
        segmentation: segmentation mask
        class_id: specific class (if None, use all tumor)
        
    Returns:
        (x, y, z) coordinates of center
    """
    if class_id is not None:
        mask = segmentation == class_id
    else:
        mask = segmentation > 0
    
    center = ndimage.center_of_mass(mask)
    return center


def get_tumor_extent(segmentation, class_id=None):
    """
    Get bounding box of tumor
    
    Args:
        segmentation: segmentation mask
        class_id: specific class
        
    Returns:
        tuple of ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    """
    if class_id is not None:
        mask = segmentation == class_id
    else:
        mask = segmentation > 0
    
    coords = np.where(mask)
    
    if len(coords[0]) == 0:
        return None
    
    extent = (
        (np.min(coords[0]), np.max(coords[0])),
        (np.min(coords[1]), np.max(coords[1])),
        (np.min(coords[2]), np.max(coords[2]))
    )
    
    return extent


def save_segmentation(segmentation, output_path, affine=None, header=None):
    """
    Save segmentation as NIfTI file
    
    Args:
        segmentation: segmentation mask
        output_path: path to save file
        affine: affine transformation matrix
        header: NIfTI header
        
    Returns:
        True if successful
    """
    try:
        if affine is None:
            affine = np.eye(4)
        
        img = nib.Nifti1Image(segmentation.astype(np.int16), affine, header)
        nib.save(img, output_path)
        
        logger.info(f"Saved segmentation to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving segmentation: {e}")
        return False


def save_metrics(metrics, output_path):
    """
    Save metrics to JSON file
    
    Args:
        metrics: dictionary of metrics
        output_path: path to save JSON
        
    Returns:
        True if successful
    """
    try:
        # Convert numpy types to native Python types
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics_serializable[key] = float(value)
            else:
                metrics_serializable[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        logger.info(f"Saved metrics to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        return False


def validate_nifti(filepath):
    """
    Validate NIfTI file
    
    Args:
        filepath: path to NIfTI file
        
    Returns:
        (is_valid, message, metadata)
    """
    try:
        img = nib.load(filepath)
        data = img.get_fdata()
        
        metadata = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'min_value': float(np.min(data)),
            'max_value': float(np.max(data)),
            'mean_value': float(np.mean(data)),
            'spacing': img.header.get_zooms()[:3],
            'orientation': nib.aff2axcodes(img.affine)
        }
        
        # Check if 3D
        if len(data.shape) < 3:
            return False, "File must be 3D volume", metadata
        
        # Check if volume is reasonable
        if np.prod(data.shape) > 1e9:
            return False, "Volume too large", metadata
        
        return True, "Valid NIfTI file", metadata
        
    except Exception as e:
        return False, f"Error loading file: {str(e)}", None


def create_report_data(mri_metadata, segmentation, metrics, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Create comprehensive report data
    
    Args:
        mri_metadata: metadata from preprocessing
        segmentation: segmentation mask
        metrics: calculated metrics
        voxel_spacing: voxel spacing
        
    Returns:
        dictionary with report data
    """
    report = {
        'input_data': {
            'original_shape': mri_metadata.get('original_shape'),
            'original_spacing': mri_metadata.get('original_spacing'),
            'preprocessing_steps': mri_metadata.get('steps_applied', [])
        },
        'tumor_characteristics': {
            'total_volume_cm3': metrics.get('total_tumor_volume_cm3', 0),
            'center_of_mass': get_tumor_center(segmentation),
            'extent': get_tumor_extent(segmentation),
            'surface_area_mm2': metrics.get('surface_area_mm2', 0)
        },
        'class_volumes': {
            'necrotic_core_cm3': metrics.get('volume_class_1_cm3', 0),
            'edema_cm3': metrics.get('volume_class_2_cm3', 0),
            'enhancing_tumor_cm3': metrics.get('volume_class_3_cm3', 0)
        },
        'performance_metrics': {}
    }
    
    # Add comparison metrics if available
    if 'dice_overall' in metrics:
        report['performance_metrics'] = {
            'dice_coefficient': metrics.get('dice_overall'),
            'iou': metrics.get('iou_overall'),
            'sensitivity': metrics.get('sensitivity_overall'),
            'specificity': metrics.get('specificity_overall')
        }
    
    return report


def format_metric_for_display(value, metric_type='percentage'):
    """
    Format metric value for display
    
    Args:
        value: metric value
        metric_type: 'percentage', 'volume', 'area'
        
    Returns:
        formatted string
    """
    if value is None:
        return "N/A"
    
    if metric_type == 'percentage':
        return f"{value * 100:.2f}%"
    elif metric_type == 'volume':
        return f"{value:.2f} cm³"
    elif metric_type == 'area':
        return f"{value:.2f} mm²"
    else:
        return f"{value:.4f}"