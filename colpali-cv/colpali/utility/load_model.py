import pprint
from typing import List, cast
import numpy as np
import os
from pdf2image import convert_from_path
from PIL import Image as PILImage

import torch
from datasets import Dataset, Features, Image, Value, load_dataset
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device


device = get_torch_device("auto")
print(f"Device used: {device}")

# Define adapter name
base_model_name = "vidore/colpaligemma-3b-pt-448-base"
adapter_name = "vidore/colpali-v1.2"

# Load model
model = ColPali.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()
model.load_adapter(adapter_name)
processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained("google/paligemma-3b-mix-448"))

if not isinstance(processor, BaseVisualRetrieverProcessor):
    raise ValueError("Processor should be a BaseVisualRetrieverProcessor")