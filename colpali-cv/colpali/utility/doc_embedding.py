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
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device

from colpali.utility.load_model import model, processor
from colpali.utility.build_dataset import images

EMBEDDING_FOLDER = "/Users/aditya.narayan/Desktop/ColPali-CV_Parsing/colpali-cv/colpali/doc_embedding"

try:
    ds: List[torch.Tensor] = torch.load(f'{EMBEDDING_FOLDER}/embeddings.pt', weights_only=True)
    print("✅ Doc Embedding Found")
except:
    print("🛠️ Initating Doc Embedding")
    # Run inference - docs
    dataloader = DataLoader(
        dataset=ListDataset[str](images),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    ds: List[torch.Tensor] = []
    for batch_doc in tqdm(dataloader, '🛠️ Creating Embedding '):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    # torch.save(ds, f'{EMBEDDING_FOLDER}/embeddings.pt')

print("✅ Doc Embedding Loaded")