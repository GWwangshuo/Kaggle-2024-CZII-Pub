# reference: https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/521786


from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from metrics.heng_utils import compute_lb, location_to_df, probability_to_location


class CZII2024Metrics:
    def __init__(self, raw_data_dir, category, valid_id, threshold):
        self.overlay_dir = (
            Path(raw_data_dir)
            / "czii-cryo-et-object-identification/train/overlay/ExperimentRuns"
        )
        self.category = category
        self.valid_id = valid_id
        self.threshold = threshold

    def __call__(self, probability: np.ndarray, valid_id: str) -> float:
        submit_df = []

        try:

            location = probability_to_location(
                probability, {k: v for k, v in zip(self.category, self.threshold)}
            )
            df = location_to_df(location)
            df.insert(loc=0, column="experiment", value=self.valid_id)
            submit_df.append(df)

            submit_df = pd.concat(submit_df)
            submit_df.insert(loc=0, column="id", value=np.arange(len(submit_df)))
            gb, lb_score = compute_lb(submit_df, self.overlay_dir)
        except Exception as e:
            gb, lb_score = 0, 0

        return gb, lb_score



class CZII2024MetricsV2:
    def __init__(self, raw_data_dir, config):
        self.overlay_dir = (
            Path(raw_data_dir)
            / "czii-cryo-et-object-identification/train/overlay/ExperimentRuns"
        )
        self.config = config

    def __call__(self, probabilities: List[np.ndarray], valid_ids: List[str], valid_id: str) -> float:
        submit_df = []
        vid_submit_df = []

        try:
            for idx, probability in enumerate(probabilities):
                vid = valid_ids[idx]
                category = self.config["category"]
                exp_name = self.config[vid]["name"]
                threshold = self.config[vid]["threshold"]

                location = probability_to_location(
                    probability, {k: v for k, v in zip(category, threshold)}
                )
                df = location_to_df(location)
                df.insert(loc=0, column="experiment", value=exp_name)
                submit_df.append(df)
                
                if vid == valid_id:
                    vid_submit_df.append(df)

            # calculate single score
            vid_submit_df = pd.concat(vid_submit_df)
            vid_submit_df.insert(loc=0, column="id", value=np.arange(len(vid_submit_df)))
            vid_gb, vid_lb_score = compute_lb(vid_submit_df, self.overlay_dir)
            
            submit_df = pd.concat(submit_df)
            submit_df.insert(loc=0, column="id", value=np.arange(len(submit_df)))
            gb, lb_score = compute_lb(submit_df, self.overlay_dir)
        except Exception as e:
            gb, lb_score = 0, 0

        return gb, lb_score, vid_gb, vid_lb_score





class CZII2024MetricsV3:
    def __init__(self, raw_data_dir, config):
        self.overlay_dir = (
            Path(raw_data_dir)
            / "czii-cryo-et-object-identification/train/overlay/ExperimentRuns"
        )
        self.config = config

    def __call__(self, probabilities: List[np.ndarray], valid_ids: List[str], valid_id: str) -> float:
        submit_df = []
        vid_submit_df = []

        try:
            for idx, probability in enumerate(probabilities):
                vid = valid_ids[idx]
                category = self.config["category"]
                exp_name = self.config[vid]["name"]
                threshold = self.config[vid]["threshold"]

                location = probability_to_location(
                    probability, {k: v for k, v in zip(category, threshold)}
                )
                df = location_to_df(location)
                df.insert(loc=0, column="experiment", value=exp_name)
                submit_df.append(df)
                
                if vid == valid_id:
                    vid_submit_df.append(df)

            # calculate single score
            vid_submit_df = pd.concat(vid_submit_df)
            vid_submit_df.insert(loc=0, column="id", value=np.arange(len(vid_submit_df)))
            vid_gb, vid_lb_score = compute_lb(vid_submit_df, self.overlay_dir)
            
            submit_df = pd.concat(submit_df)
            submit_df.insert(loc=0, column="id", value=np.arange(len(submit_df)))
            gb, lb_score = compute_lb(submit_df, self.overlay_dir)
        except Exception as e:
            gb, lb_score, vid_gb, vid_lb_score = 0, 0, 0, 0

        return vid_gb, vid_lb_score, gb, lb_score, 




