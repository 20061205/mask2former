"""
Model loading and semantic segmentation using Mask2Former.
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from config import MODEL_NAME


def load_model(model_name: str = MODEL_NAME):
    """Load Mask2Former model and processor."""
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return processor, model, device


def segment_image(image_rgb: Image.Image, processor, model, device) -> np.ndarray:
    """
    Run semantic segmentation on an RGB PIL image.

    Returns
    -------
    segmentation_map : np.ndarray  (H, W) with integer class IDs
    """
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[image_rgb.size[::-1]],
    )[0]

    return result.cpu().numpy()


def get_label_map(model):
    """Return the id→label dict from the model config."""
    return model.config.id2label
