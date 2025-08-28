# logging_setup.py
import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging() -> logging.Logger:
    """
    Configure un logger 'ai-server' avec :
      - sortie console
      - fichier rotatif ~/ai-server/logs/server.log
      - intégration des logs uvicorn
    Appelle cette fonction une seule fois (dans main.py).
    """
    log_dir = os.getenv("LOG_DIR", os.path.expanduser("~/ai-server/logs"))
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("ai-server")
    if logger.handlers:
        return logger  # déjà configuré

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(process)d:%(threadName)s] %(message)s"
    )

    # Fichier rotatif
    file_path = os.path.join(log_dir, "server.log")
    fh = RotatingFileHandler(file_path, maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console (captée par journalctl en service systemd)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Harmoniser les loggers uvicorn avec notre config
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        l = logging.getLogger(name)
        l.handlers = logger.handlers
        l.setLevel(level)
        l.propagate = False

    return logger
