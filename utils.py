import random
from datetime import datetime
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def haversine(coordinate, coordinates_array):
    lon1, lat1 = coordinate
    lon2_array, lat2_array = coordinates_array[:, 0], coordinates_array[:, 1]
    lon1, lat1, lon2_array, lat2_array = map(np.radians, [lon1, lat1, lon2_array, lat2_array])

    dlon = lon2_array - lon1
    dlat = lat2_array - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2_array) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    r = 6371.0
    return c * r
