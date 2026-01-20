import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
import dill


def save_object(file_path, obj):
    """Saves a Python object to a file using pickle.

    Args:
        file_path (str): The path where the object should be saved.
        obj: The Python object to be saved.

    Raises:
        CustomException: If there is an error during the saving process.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        import pickle
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        logging.info("Exception occurred while saving object")
        raise CustomException(e, sys)