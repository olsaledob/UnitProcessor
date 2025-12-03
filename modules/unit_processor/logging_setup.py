import logging
import os
from datetime import datetime

def setup_unit_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"unitproc_log_{timestamp}.log")

    logger = logging.getLogger("unit_processor")
    logger.setLevel(getattr(logging, log_level.upper()))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s.%(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level.upper()))
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(getattr(logging, log_level.upper()))
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info("Unit Processor logging initialised")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {log_level}")

    return logger