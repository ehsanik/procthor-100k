import gzip
import os
from tqdm import tqdm
import prior
import urllib.request

try:
    from prior import LazyJsonDataset
except:
    raise ImportError("Please update the prior package (pip install --upgrade prior).")

def load_dataset() -> prior.DatasetDict:
    """Load the houses dataset."""
    data = {}
    for split, size in zip(("train","val","test"),(100_000, 1_000, 1_000)):
        if not f"procthor100k_balanced_partitionedassets_{split}.jsonl.gz" in os.listdir('./'):
                url = f"https://prior-datasets.s3.us-east-2.amazonaws.com/ilearn-dataset/procthor_100k_balanced_{split}.jsonl.gz"
                urllib.request.urlretrieve(url,'./procthor_100k_balanced_{}.jsonl.gz'.format(split))
        with gzip.open(f"procthor100k_balanced_partitionedassets_{split}.jsonl.gz", "r") as f:
            houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
        data[split] = LazyJsonDataset(
            data=houses, dataset="procthor-100k", split=split
        )
    return prior.DatasetDict(**data)
