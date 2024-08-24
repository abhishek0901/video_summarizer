'''
================================================

Helper methods to read and write a video.

Author : Abhishek Srivastava

================================================
'''
import cv2
import numpy as np
import scipy
from helper_methods.wrappers import measure_execution_time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='tmp/run.log', encoding='utf-8', level=logging.INFO)

@measure_execution_time
def get_frames_and_gt_from_video(video_path: str, return_ground_truth: bool = False) -> np.array:
    """
        Description
        ===========
        Uses opencv to read frames from a valid mp4 video and convert it into RGB features.
        Due to frame redundency it only pickes every 5th frame. Also returns ground truth score
        corresponding to the frame. Null if False.

        Example
        =======
        Let's say you have a video with 512 frames then this will convert the video into frame of shape -> 512 X WIDTH X HEIGHT X 3
    """
    logger.info(f"Collecting features for {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    gt_scores = None
    frame_num = 0

    all_frame_score = None
    if return_ground_truth:
        all_frame_score = scipy.io.loadmat(video_path.replace("videos/", "GT/").replace(".mp4", ".mat"))['gt_score']
        gt_scores = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        if frame_num%5 == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            gt_scores.append(all_frame_score[frame_num])
        frame_num += 1
    cap.release()

    num_frames = len(frames)
    frame_shape = (0,0,0) # WIDTH X HEIGHT X CHANNEL
    if num_frames != 0:
        frame_shape = frames[0].shape
    
    logger.info("feature extraction complete.", extra={
        'tags':{
            num_frames : num_frames,
            frame_shape : frame_shape
        }
    })

    return np.array(frames), np.array(gt_scores)


if __name__ == "__main__":
    frames, gt_scores = get_frames_and_gt_from_video("/home/paperspace/src/video_summarizer/data/datasets/SumMe/videos/Air_Force_One.mp4", return_ground_truth=True)
    print(len(frames), len(gt_scores))
    print(frames[0].shape)