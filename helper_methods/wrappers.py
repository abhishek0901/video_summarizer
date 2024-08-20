import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='tmp/run.log', encoding='utf-8', level=logging.INFO)

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f'Execution time: {end_time - start_time} seconds')
        return result
    return wrapper