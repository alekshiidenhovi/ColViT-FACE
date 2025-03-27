import faiss
import numpy as np


class VectorIndex:
    def __init__(
        self,
        vector_dim: int,
        n_centroids: int = 100,
        n_probe: int = 10,
        n_subvectors: int = 8,
        n_bits: int = 8,
    ):
        """Initialize a vector index for efficient similarity search.

        Args:
            vector_dim: Dimensionality of the input vectors
            n_centroids: Number of Voronoi cells (clusters) for the coarse quantizer.
            n_probe: Number of nearest cells to search during query.
            n_subvectors: Number of subvectors to use for Product Quantization.
            n_bits: Number of bits used for Product Quantization encoding.
        """

        if n_centroids < 1:
            raise ValueError("n_centroids must be greater than 0")
        if n_subvectors < 1:
            raise ValueError("n_subvectors must be greater than 0")
        if n_bits < 1:
            raise ValueError("n_bits must be greater than 0")
        if n_probe < 1:
            raise ValueError("n_probe must be greater than 0")
        if n_probe > n_centroids:
            raise ValueError("n_probe must be less than or equal to n_centroids")
        if vector_dim % n_subvectors != 0:
            raise ValueError("vector_dim must be divisible by n_subvectors")

        self.vector_dim = vector_dim
        self.quantizer = faiss.IndexFlatL2(vector_dim)
        self.index = faiss.IndexIVFPQ(
            self.quantizer, vector_dim, n_centroids, n_subvectors, n_bits
        )
        self.index.nprobe = n_probe

    def add(self, vectors: np.ndarray, vector_ids: np.ndarray):
        """Add vectors to the index.

        Args:
            vectors: Numpy array of shape (n_vectors, vector_dim) containing the vectors to add. Each vector must have the same dimensionality as specified in vector_dim.
            ids: Numpy array of shape (n_vectors,) containing the IDs for the vectors.

        Raises:
            ValueError: If vectors do not match the expected dimensionality or if vector_ids do not match the expected length.
        """
        if vectors.shape[1] != self.vector_dim:
            raise ValueError(
                f"Expected vectors with dimension {self.vector_dim}, "
                f"but got vectors with dimension {vectors.shape[1]}"
            )

        if vector_ids.shape[0] != vectors.shape[0]:
            raise ValueError(
                f"Expected vector_ids with length {vectors.shape[0]}, "
                f"but got vector_ids with length {vector_ids.shape[0]}"
            )

        if not np.issubdtype(vector_ids.dtype, np.int64):
            raise ValueError("vector_ids must be 64-bit integers")

        self.index.train(vectors)
        self.index.add_with_ids(vectors, vector_ids)

    def search(self, vectors: np.ndarray, k: int):
        """Search for the k nearest neighbors to the input vectors.

        Args:
            vectors: Numpy array of shape (n_vectors, vector_dim) containing the vectors to search.
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of two arrays:
                - Distances: Numpy array of shape (n_vectors, k) containing the distances to the k nearest neighbors.
                - Indices: Numpy array of shape (n_vectors, k) containing the indices of the k nearest neighbors.

        Raises:
            ValueError: If vectors do not match the expected dimensionality.
        """
        if vectors.shape[1] != self.vector_dim:
            raise ValueError(
                f"Expected vectors with dimension {self.vector_dim}, "
                f"but got vectors with dimension {vectors.shape[1]}"
            )

        return self.index.search(vectors, k)
