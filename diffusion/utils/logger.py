import logging
import os
import torch.distributed as dist
from datetime import datetime
from .dist_utils import is_local_master
from mmcv.utils.logging import logger_initialized


def get_root_logger(log_file=None, log_level=logging.INFO, name='PixArt'):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str): logger name
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    if log_file is None:
        log_file = '/dev/null'
    return get_logger(name=name, log_file=log_file, log_level=log_level)


def get_logger(name, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    logger.propagate = False  # disable root logger to avoid duplicate logging

    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    # only rank0 for each node will print logs
    log_level = log_level if is_local_master() else logging.ERROR
    logger.setLevel(log_level)

    logger_initialized[name] = True

    return logger

def rename_file_with_creation_time(file_path):
    # 获取文件的创建时间
    creation_time = os.path.getctime(file_path)
    creation_time_str = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d_%H-%M-%S')

    # 构建新的文件名
    dir_name, file_name = os.path.split(file_path)
    name, ext = os.path.splitext(file_name)
    new_file_name = f"{name}_{creation_time_str}{ext}"
    new_file_path = os.path.join(dir_name, new_file_name)

    # 重命名文件
    os.rename(file_path, new_file_path)
    print(f"File renamed to: {new_file_path}")
