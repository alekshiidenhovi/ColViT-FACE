import kagglehub
from time import time


def download_casia_webface() -> None:
    try:
        start_time: float = time()
        path: str = kagglehub.dataset_download(
            "cybersimar08/casia-face-dataset", force_download=True
        )
        spent_time: float = time() - start_time
        print(
            f"CASIA WebFace dataset downloaded successfully in {spent_time} seconds.\nDataset path: {path}"
        )
    except Exception as e:
        print(f"Unexpected error occurred while downloading CASIA WebFace dataset: {e}")


if __name__ == "__main__":
    download_casia_webface()
