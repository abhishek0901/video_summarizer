'''
================================================

Build Dataset class to hold and serve video data.

Author : Abhishek Srivastava

Docs
1. Data is downloaded from https://zenodo.org/records/4884870
================================================
'''
from typing import Tuple

import os
import torch
from tqdm import tqdm
import numpy as np
from config import CONFIG
from torch.utils.data import Dataset
import logging
from helper_methods.video_io import get_frames_and_gt_from_video

logger = logging.getLogger(__name__)
logging.basicConfig(filename='tmp/run.log', encoding='utf-8', level=logging.INFO)

class Video:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    """
        Description
        -----------
        Unit class to hold video features and ground truth value.
    """
    def __init__(self, features, gtscores) -> None:
        self.features = features
        self.gtscores = gtscores


class VideoDataSet(Dataset):
    """
        Description
        -----------
        This class holds video data stored in hdf5 format.

        Arguments
        ---------

        video_directory_path : path to video directory.

        Returns
        -------
        Returns DataSet object.

    """

    def __init__(self, video_directory_path: str) -> None:
        """
            - Read video from path
            - store all videos and their output in an array
        """
        all_frames = []

        logger.info("loading video data.")
        for file in tqdm(os.listdir(video_directory_path)):
            if file.endswith(".mp4"):
                file_name = os.path.join(video_directory_path, file)
                features, gtscores = get_frames_and_gt_from_video(file_name, return_ground_truth=True)
                all_frames.append(
                    Video(features=features, gtscores=gtscores)
                )
        
        self.all_frames = all_frames
        logger.info("video data loaded.")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        video = self.all_frames[index]
        features = video.features
        gtscores = video.gtscores

        # Reshape to Make X as (T,3,H,W)
        features = np.moveaxis(features, -1,1)

        return torch.from_numpy(features).float().to(CONFIG.DEVICE), torch.from_numpy(gtscores).to(CONFIG.DEVICE)

if __name__ == '__main__':
    video_path = "/home/paperspace/src/video_summarizer/data/datasets/SumMe/videos"
    vd = VideoDataSet(video_path)
    feature, gtscore  = vd[0]
    logger.info(feature.shape)