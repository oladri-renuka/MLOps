import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api_requests.log"),  # saves logs
        logging.StreamHandler()  # prints to console
    ]
)

logger = logging.getLogger(__name__)
