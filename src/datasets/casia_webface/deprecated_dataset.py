import os
import numpy as np
import io
import typing as T
from torch.utils.data import Dataset
from PIL import Image
from datasets.TSVLoader import TSVLoader
from datasets.RecordReader import RecordReader
from pathlib import Path
from torchvision import transforms


class CASIATripletDataset(Dataset):
    """
    A PyTorch Dataset that reads .rec, .idx, and .lst from CASIA (or similar) format using MXNet's recordio.
    """

    def __init__(
        self,
        idx_file: str,
        rec_file: str,
        lst_file: str,
        transform: T.Optional[transforms.Compose] = None,
        subset_indices: T.Optional[T.List[int]] = None,
    ):
        """
        Args:
            idx_file (str): Path to .idx
            rec_file (str): Path to .rec
            lst_file (str): Path to .lst
            transform (callable, optional): Optional transform to be applied on a sample.
            subset_indices (list[int], optional): List of row indices to use for this dataset split.
        """
        super().__init__()
        self.idx_file = idx_file
        self.rec_file = rec_file
        self.lst_file = lst_file
        self.transform = transform
        self.samples = []

        tsv_loader = TSVLoader(file_path=Path(self.lst_file))
        lines = tsv_loader.load_lines()
        for idx, line in enumerate(lines):
            parts: list[str] = tsv_loader.split_line(line)
            image_path_str, label_str = parts[1], parts[2]
            label = int(label_str)
            image_path_end = os.path.splitext(os.path.basename(image_path_str))
            image_id: int = int(image_path_end[0])
            self.samples.append((image_id, label, idx))

        if subset_indices is not None:
            self.samples = [self.samples[i] for i in subset_indices]

        self.record = RecordReader(idx_file, rec_file)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, label, rec_idx = self.samples[idx]

        # record.read_idx expects 1-based index in some datasets.
        # If your .lst file starts from 0, you might need +1.
        # This depends on how the dataset is built.
        # Try first as is, if you get None, do rec_index+1
        s = self.record.read_idx(rec_idx)
        if s is None:
            # fallback
            s = self.record.read_idx(rec_idx + 1)
            if s is None:
                raise ValueError(f"Could not read record at index {rec_idx}")

        img_data = s[20:]
        img_pil = Image.open(io.BytesIO(img_data))
        img_pil = img_pil.convert("RGB")
        if self.transform:
            img_pil = self.transform(img_pil)

        return img_pil, label


if __name__ == "__main__":
    import plotly.graph_objects as go

    dataset_path = "/teamspace/studios/this_studio/ColViT-FACE/data"
    idx_file = os.path.join(dataset_path, "train.idx")
    rec_file = os.path.join(dataset_path, "train.rec")
    lst_file = os.path.join(dataset_path, "train.lst")

    dataset = CASIATripletDataset(
        idx_file=idx_file, rec_file=rec_file, lst_file=lst_file
    )

    # Get first image
    img, label = dataset[0]

    # Convert PIL image to numpy array
    img_array = np.array(img)

    fig = go.Figure(data=go.Image(z=img_array))

    fig.update_layout(
        title=f"Label: {label}", width=400, height=400, margin=dict(l=0, r=0, t=30, b=0)
    )

    fig.show()

    print(f"Dataset size: {len(dataset)}")
    print(f"Image size: {img.size}")
    print(f"Label: {label}")
