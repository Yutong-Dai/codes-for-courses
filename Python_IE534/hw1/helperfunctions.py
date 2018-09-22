import logging


def create_log(file_name="task.log", log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    handler = logging.FileHandler(file_name)
    handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


if __name__ == "__main__":
    mylogger = create_log(file_name="task.log", log_level=logging.INFO)
    mylogger.info('Hello world!')
    mylogger.debug("Something is wrong!")
