import logging

# Configure logging format
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG  # Default logging level (can be adjusted)
)

# Create a logger instance
logger = logging.getLogger(__name__)