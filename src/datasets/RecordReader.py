import os


class RecordReader:
    """A pure Python implementation to read MXNet RecordIO format."""

    def __init__(self, idx_file: str, rec_file: str):
        self.rec_file = rec_file
        self.rec_handle = open(rec_file, "rb")
        self.file_size = os.path.getsize(rec_file)
        self.idx_map = {}
        with open(idx_file, "r") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        idx = int(parts[0])
                        pos = int(parts[1])
                        self.idx_map[idx] = pos

    def read_idx(self, idx):
        if idx not in self.idx_map:
            return None

        self.rec_handle.seek(self.idx_map[idx])
        current_pos = self.idx_map[idx]
        next_pos = self.idx_map[idx + 1] if idx + 1 in self.idx_map else self.file_size

        record_size = next_pos - current_pos
        print(
            f"Current pos: {current_pos}, next pos: {next_pos}, record size: {record_size}"
        )
        record_data = self.rec_handle.read(record_size)
        return record_data

    def close(self):
        self.rec_handle.close()