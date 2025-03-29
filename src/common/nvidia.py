import subprocess
from common.logger import logger


def get_gpu_info_from_nvidia_smi():
    """Get detailed GPU information using nvidia-smi command.

    This function executes the nvidia-smi command to retrieve information about installed
    NVIDIA GPUs including their names, total memory, and driver versions.

    Returns
    -------
    tuple or None
        If successful, returns a tuple containing strings with GPU information, where each
        string contains comma-separated values in the format:
        "GPU name, total memory (MiB), driver version"
        Returns None if nvidia-smi command fails or is not found.
    """

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=True,  # Raises CalledProcessError if return code is non-zero
        )

        gpu_info = []
        for line in result.stdout.strip().split("\n"):
            if line:
                gpu_info.append(line.strip())
        return tuple(gpu_info) if gpu_info else None

    except FileNotFoundError:
        logger.warning(
            "nvidia-smi command not found. Please ensure NVIDIA drivers are installed correctly."
        )
        return None
    except subprocess.SubprocessError as e:
        logger.warning(f"Failed to execute nvidia-smi command: {str(e)}")
        return None
