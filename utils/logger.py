import logging

def setup_logger():
    """Настройка логгера."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("parser.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()