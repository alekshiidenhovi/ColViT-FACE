from pathlib import Path

class TSVLoader():
    """
    Helper class for processing tab-separated value (TSV) files.
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
    
    def load_lines(self, sep="\n", mode="r") -> list[str]:
        with open(str(self.file_path), mode=mode) as f:
            lines = f.read().strip().split(sep)
        return lines

    def split_line(self, line: str, sep="\t") -> list[str]:
        parts: list[str] = line.strip().split(sep=sep)
        return parts
