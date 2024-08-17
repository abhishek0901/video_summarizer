'''
================================================

Build Dataset class to hold and serve video data.

Author : Abhishek Srivastava

================================================
'''
import h5py
from torch.utils.data import Dataset
from typing import Any
import logging

logger = logging.getLogger(__name__)


class VideoDataSet(Dataset):
    """
        Description
        -----------
        This class holds video data stored in hdf5 format.

        Arguments
        ---------

        video_path : path to video file in hdf5 format.

        Returns
        -------
        Returns DataSet object.

    """

    def __init__(self, video_path: str) -> None:
        """
            1. Read video from path
            2. store all videos and their output in an array
        """
        



    def __len__(self):
        pass

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)


if __name__ == '__main__':
    print("Hello World!")