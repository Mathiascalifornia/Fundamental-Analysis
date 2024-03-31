import pickle
from typing import Any

class PickleLoaderAndSaviour:

    @staticmethod
    def load_pickle_object(file_path) -> Any:
        """
        Load an object from a pickle file
        """
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def save_pickle_object(obj, file_path):
        """
        Save an object to a pickle file
        """
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)