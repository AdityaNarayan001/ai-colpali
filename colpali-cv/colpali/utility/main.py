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

from colpali.utility.build_dataset import filenames
from colpali.utility.load_model import processor, model
from colpali.utility.doc_embedding import ds


def main(queries):
    queries = [queries]

    # Run inference - queries
    dataloader = DataLoader(
        dataset=ListDataset[str](queries),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_queries(x),
    )
    qs: List[torch.Tensor] = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    # Run scoring
    scores = processor.score(qs, ds).cpu().numpy()

    top_n = 3
    top_n_indices = np.argsort(scores, axis=1)[:, -top_n:]  # Get the last 3 indices (highest scores)
    top_n_indices = top_n_indices[:, ::-1]

    infer_dict = {}

    for i in range(len(queries)):
        for rank, idx in enumerate(top_n_indices[i]):
            infer_dict[str(rank+1)] = {}
            infer_dict[str(rank+1)]["rank"] = str(rank+1)
            infer_dict[str(rank+1)]["filename"] = filenames[idx]
            infer_dict[str(rank+1)]["score"] = str(scores[i][idx])
            # print(f"  Rank {rank + 1}: Filename: {filenames[idx]}, Score: {scores[i][idx]}")

    queries = []

    return infer_dict


if __name__ == "__main__":
    main()