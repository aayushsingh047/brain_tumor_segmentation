"""
Brain Tumor Segmentation – Model Loader
========================================
Loads the trained 3D U-Net and exposes a unified predict() interface.

Usage:
    model = load_model("checkpoints/model.pth")          # real weights
    model = load_model(use_mock=True)                     # demo / no GPU
    seg, probs = model.predict(input_array)
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────

def _build_unet(in_channels: int = 4, out_channels: int = 4) -> nn.Module:
    """
    3D U-Net via MONAI.

    Encoder path  : channels 16 → 32 → 64 → 128 → 256 (bottleneck)
                    each stage doubles channels, stride-2 downsampling
    Decoder path  : mirrors encoder with transposed convolutions + skip concat
    Residual units: 2 per block for better gradient flow
    Batch Norm    : improves training stability for 3D medical data
    Dropout 0.1   : mild regularisation

    Input  : (B, 4, 128, 128, 128)  → 4 MRI modalities
    Output : (B, 4, 128, 128, 128)  → logits for 4 classes
    """
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=0.1,
    )


# ─────────────────────────────────────────────────────────
# Real model wrapper
# ─────────────────────────────────────────────────────────

class BrainTumorSegmentationModel:
    """
    Wraps the 3D U-Net for inference.

    Args:
        device           : torch.device or None (auto-detect)
        checkpoint_path  : optional path to model.pth
    """

    CLASS_NAMES = ["Background", "Necrotic Core", "Edema", "Enhancing Tumor"]

    def __init__(self, device=None, checkpoint_path=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = _build_unet().to(self.device)
        self.model.eval()

        if checkpoint_path:
            self.load_weights(checkpoint_path)

        logger.info(f"Model ready on {self.device} | "
                    f"params: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M")

    # ── Weight loading ────────────────────────────────────

    def load_weights(self, checkpoint_path: str):
        """
        Load model weights from a .pth checkpoint.

        Supports two formats:
          1. Full checkpoint dict with key 'model_state_dict'  (saved by train.py)
          2. Bare state-dict  (saved with torch.save(model.state_dict(), path))
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Train the model first: python train.py --data_dir <path>"
            )

        ckpt = torch.load(path, map_location=self.device)

        # Determine format
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            epoch = ckpt.get("epoch", "?")
            metrics = ckpt.get("val_metrics", {})
            logger.info(f"Loading checkpoint epoch={epoch} | "
                        f"val_dice={metrics.get('mean_dice', 'N/A')}")
        else:
            state_dict = ckpt   # bare state dict
            logger.info("Loading bare state dict")

        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"Weights loaded from {path}")

    # ── Inference ─────────────────────────────────────────

    def _prepare_input(self, input_data: np.ndarray) -> torch.Tensor:
        """
        Accept (H, W, D) single-channel or (4, H, W, D) multi-channel.
        Returns batched tensor (1, 4, D, H, W) on the correct device.
        """
        if input_data.ndim == 3:
            # Single channel → replicate to 4 channels
            input_data = np.stack([input_data] * 4, axis=0)
        # shape is now (4, H, W, D) – add batch dim
        return torch.from_numpy(input_data).float().unsqueeze(0).to(self.device)

    def predict(self,
                input_data: np.ndarray,
                confidence_threshold: float = 0.5) -> tuple:
        """
        Standard inference (full volume in one forward pass).

        Args:
            input_data           : (4, D, H, W) or (D, H, W) numpy array
            confidence_threshold : voxels whose max softmax prob falls below
                                   this threshold are set to background (0)

        Returns:
            segmentation_mask : (D, H, W) int array  (class indices 0-3)
            probabilities     : (4, D, H, W) float array
        """
        with torch.no_grad():
            tensor = self._prepare_input(input_data)
            logits = self.model(tensor)                     # (1, 4, D, H, W)
            probs  = torch.softmax(logits, dim=1)

            max_prob, pred = torch.max(probs, dim=1)
            pred[max_prob < confidence_threshold] = 0       # low-confidence → bg

        return pred.squeeze(0).cpu().numpy(), probs.squeeze(0).cpu().numpy()

    def predict_sliding_window(self,
                               input_data: np.ndarray,
                               roi_size=(128, 128, 128),
                               overlap: float = 0.5) -> tuple:
        """
        Sliding-window inference for volumes larger than roi_size.
        Recommended when input is not already cropped to 128³.

        Returns:
            segmentation_mask : (D, H, W) int array
            probabilities     : (4, D, H, W) float array
        """
        with torch.no_grad():
            tensor = self._prepare_input(input_data)
            logits = sliding_window_inference(
                inputs      = tensor,
                roi_size    = roi_size,
                sw_batch_size = 2,
                predictor   = self.model,
                overlap     = overlap,
                mode        = "gaussian",
            )
            probs = torch.softmax(logits, dim=1)
            pred  = torch.argmax(probs, dim=1)

        return pred.squeeze(0).cpu().numpy(), probs.squeeze(0).cpu().numpy()


# ─────────────────────────────────────────────────────────
# Mock model (demo / fallback)
# ─────────────────────────────────────────────────────────

class MockSegmentationModel:
    """
    Synthetic segmentation model for demonstration when no trained
    weights are available. Uses intensity-based heuristics to generate
    plausible-looking (but not clinically valid) tumour masks.
    """

    CLASS_NAMES = ["Background", "Necrotic Core", "Edema", "Enhancing Tumor"]

    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
        logger.warning(
            "Using MOCK model – predictions are synthetic, NOT clinical quality."
        )

    def predict(self, input_data: np.ndarray,
                confidence_threshold: float = 0.5) -> tuple:
        """Generate synthetic segmentation via intensity thresholding."""
        from scipy.ndimage import gaussian_filter, binary_dilation

        data = input_data[0] if input_data.ndim == 4 else input_data
        norm = (data - data.mean()) / (data.std() + 1e-8)
        smoothed = gaussian_filter(norm, sigma=2)

        seg = np.zeros_like(data, dtype=np.int32)
        seg[binary_dilation(smoothed > 2.0, iterations=2)]              = 3
        seg[binary_dilation((smoothed > 1.5) & (smoothed <= 2.0), iterations=3) & (seg == 0)] = 1
        seg[binary_dilation((smoothed > 1.0) & (smoothed <= 1.5), iterations=4) & (seg == 0)] = 2

        probs = np.zeros((4,) + data.shape, dtype=np.float32)
        for i in range(4):
            probs[i][seg == i] = 0.7 + np.random.rand() * 0.25
        probs /= probs.sum(axis=0, keepdims=True) + 1e-8

        return seg, probs

    def predict_sliding_window(self, input_data, **kwargs):
        return self.predict(input_data)[0], None


# ─────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────

def load_model(checkpoint_path: str = None,
               use_mock: bool = False,
               device=None):
    """
    Load and return a segmentation model.

    Priority:
      1. If use_mock=True → return MockSegmentationModel
      2. If checkpoint_path provided and file exists → load real model
      3. If checkpoint_path is None but checkpoints/model.pth exists → auto-load
      4. Fallback to MockSegmentationModel with a warning

    Args:
        checkpoint_path : explicit path to .pth file
        use_mock        : force mock model
        device          : torch.device

    Returns:
        model instance with .predict() and .predict_sliding_window()
    """
    if use_mock:
        return MockSegmentationModel(device=device)

    # Auto-discover checkpoint
    if checkpoint_path is None:
        candidates = [
            Path("checkpoints/model.pth"),
            Path("checkpoints/best_model.pth"),
            Path("model.pth"),
        ]
        for c in candidates:
            if c.exists():
                checkpoint_path = str(c)
                logger.info(f"Auto-discovered checkpoint: {checkpoint_path}")
                break

    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            model = BrainTumorSegmentationModel(
                device=device, checkpoint_path=checkpoint_path)
            return model
        except Exception as e:
            logger.error(f"Failed to load checkpoint ({e}). Falling back to mock.")

    logger.warning(
        "No trained model found. Using mock model.\n"
        "  → Train first: python train.py --data_dir <brats_path>\n"
        "  → Then place checkpoints/model.pth in the project root."
    )
    return MockSegmentationModel(device=device)


def get_model_info() -> dict:
    """Return static metadata about the model architecture."""
    return {
        "architecture"    : "3D U-Net (MONAI)",
        "framework"       : "PyTorch + MONAI",
        "input_channels"  : 4,
        "output_classes"  : 4,
        "class_names"     : ["Background", "Necrotic Core", "Edema", "Enhancing Tumor"],
        "input_shape"     : "(B, 4, 128, 128, 128)",
        "encoder_channels": "(16, 32, 64, 128, 256)",
        "parameters"      : "~31M",
        "training_dataset": "BraTS (Brain Tumor Segmentation Challenge)",
        "loss_function"   : "Combined Dice + CrossEntropy",
        "optimizer"       : "Adam (lr=0.001, weight_decay=1e-5)",
        "scheduler"       : "CosineAnnealingLR",
    }
