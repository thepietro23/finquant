import logging
import os

_LOGGERS = {}


def get_logger(name, log_dir='logs'):
    """Get or create a named logger with file + console output.

    Usage:
        logger = get_logger('data_pipeline')
        logger.info('Download started')
        logger.error('Something failed')
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers on re-import
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

        fh = logging.FileHandler(os.path.join(log_dir, f'{name}.log'), encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    _LOGGERS[name] = logger
    return logger
