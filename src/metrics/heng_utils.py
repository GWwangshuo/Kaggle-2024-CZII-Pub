import json

import cc3d
import numpy as np
import pandas as pd
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


PARTICLE = [
    {
        "name": "apo-ferritin",
        "difficulty": "easy",
        "pdb_id": "4V1W",
        "label": 1,
        "color": [0, 255, 0, 0],
        "radius": 60,
        "map_threshold": 0.0418,
    },
    {
        "name": "beta-amylase",
        "difficulty": "ignore",
        "pdb_id": "1FA2",
        "label": 2,
        "color": [0, 0, 255, 255],
        "radius": 65,
        "map_threshold": 0.035,
    },
    {
        "name": "beta-galactosidase",
        "difficulty": "hard",
        "pdb_id": "6X1Q",
        "label": 3,
        "color": [0, 255, 0, 255],
        "radius": 90,
        "map_threshold": 0.0578,
    },
    {
        "name": "ribosome",
        "difficulty": "easy",
        "pdb_id": "6EK0",
        "label": 4,
        "color": [0, 0, 255, 0],
        "radius": 150,
        "map_threshold": 0.0374,
    },
    {
        "name": "thyroglobulin",
        "difficulty": "hard",
        "pdb_id": "6SCJ",
        "label": 5,
        "color": [0, 255, 255, 0],
        "radius": 130,
        "map_threshold": 0.0278,
    },
    {
        "name": "virus-like-particle",
        "difficulty": "easy",
        "pdb_id": "6N4V",
        "label": 6,
        "color": [0, 0, 0, 255],
        "radius": 135,
        "map_threshold": 0.201,
    },
]

PARTICLE_NAME = ["none"] + [PARTICLE[i]["name"] for i in range(6)]


def probability_to_location(probability, threshold):
    _, D, H, W = probability.shape

    location = {}
    for p in PARTICLE:
        p = dotdict(p)
        l = p.label

        cc, P = cc3d.connected_components(
            probability[l] > threshold[p.name], return_N=True
        )
        stats = cc3d.statistics(cc)
        zyx = stats["centroids"][1:] * 10
        xyz = np.ascontiguousarray(zyx[:, ::-1])
        location[p.name] = xyz

    return location


def location_to_df(location):
    location_df = []
    for p in PARTICLE:
        p = dotdict(p)
        xyz = location[p.name]
        if len(xyz) > 0:
            df = pd.DataFrame(data=xyz, columns=["x", "y", "z"])
            # df.loc[:,'particle_type']= p.name
            df.insert(loc=0, column="particle_type", value=p.name)
            location_df.append(df)
    location_df = pd.concat(location_df)
    return location_df


def read_one_truth(id, overlay_dir):
    location = {}

    json_dir = f"{overlay_dir}/{id}/Picks"
    for p in PARTICLE_NAME[1:]:
        json_file = f"{json_dir}/{p}.json"

        with open(json_file, "r") as f:
            json_data = json.load(f)

        num_point = len(json_data["points"])
        loc = np.array(
            [
                list(json_data["points"][i]["location"].values())
                for i in range(num_point)
            ]
        )
        location[p] = loc

    return location


def do_one_eval(truth, predict, threshold):
    P = len(predict)
    T = len(truth)

    if P == 0:
        hit = [[], []]
        miss = np.arange(T).tolist()
        fp = []
        metric = [P, T, len(hit[0]), len(miss), len(fp)]
        return hit, fp, miss, metric

    if T == 0:
        hit = [[], []]
        fp = np.arange(P).tolist()
        miss = []
        metric = [P, T, len(hit[0]), len(miss), len(fp)]
        return hit, fp, miss, metric

    # ---
    distance = predict.reshape(P, 1, 3) - truth.reshape(1, T, 3)
    distance = distance**2
    distance = distance.sum(axis=2)
    distance = np.sqrt(distance)
    p_index, t_index = linear_sum_assignment(distance)

    valid = distance[p_index, t_index] <= threshold
    p_index = p_index[valid]
    t_index = t_index[valid]
    hit = [p_index.tolist(), t_index.tolist()]
    miss = np.arange(T)
    miss = miss[~np.isin(miss, t_index)].tolist()
    fp = np.arange(P)
    fp = fp[~np.isin(fp, p_index)].tolist()

    metric = [P, T, len(hit[0]), len(miss), len(fp)]  # for lb metric F-beta copmutation
    return hit, fp, miss, metric


def compute_lb(submit_df, overlay_dir):
    valid_id = list(submit_df["experiment"].unique())
    # print(valid_id)

    eval_df = []
    for id in valid_id:
        truth = read_one_truth(
            id, overlay_dir
        )  # =f'{valid_dir}/overlay/ExperimentRuns')
        id_df = submit_df[submit_df["experiment"] == id]
        for p in PARTICLE:
            p = dotdict(p)
            # print("\r", id, p.name, end="", flush=True)
            xyz_truth = truth[p.name]
            xyz_predict = id_df[id_df["particle_type"] == p.name][
                ["x", "y", "z"]
            ].values

            hit, fp, miss, metric = do_one_eval(xyz_truth, xyz_predict, p.radius * 0.5)
            eval_df.append(
                dotdict(
                    id=id,
                    particle_type=p.name,
                    P=metric[0],
                    T=metric[1],
                    hit=metric[2],
                    miss=metric[3],
                    fp=metric[4],
                )
            )
    # print("")
    eval_df = pd.DataFrame(eval_df)
    gb = eval_df.groupby("particle_type").agg("sum").drop(columns=["id"])
    gb.loc[:, "precision"] = gb["hit"] / gb["P"]
    gb.loc[:, "precision"] = gb["precision"].fillna(0)
    gb.loc[:, "recall"] = gb["hit"] / gb["T"]
    gb.loc[:, "recall"] = gb["recall"].fillna(0)
    gb.loc[:, "f-beta4"] = (
        17 * gb["precision"] * gb["recall"] / (16 * gb["precision"] + gb["recall"])
    )
    gb.loc[:, "f-beta4"] = gb["f-beta4"].fillna(0)

    gb = gb.sort_values("particle_type").reset_index(drop=False)
    # https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895
    gb.loc[:, "weight"] = [1, 0, 2, 1, 2, 1]
    lb_score = (gb["f-beta4"] * gb["weight"]).sum() / gb["weight"].sum()
    return gb, lb_score
