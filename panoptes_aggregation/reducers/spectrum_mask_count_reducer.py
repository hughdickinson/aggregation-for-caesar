from collections import OrderedDict
import numpy as np
import itertools

from panoptes_aggregation.reducers.reducer_wrapper import reducer_wrapper


def extract_bbox_ids(extract):
    bbox_ids = []
    for frame, frame_data in extract["data"].items():
        if "version" in frame:
            continue
        bbox_ids.extend(frame_data["bbox_id"])
        print(bbox_ids)
    return bbox_ids


@reducer_wrapper()
def spectrum_mask_count_reducer(data_list, **kwargs):
    reductions = OrderedDict()
    bbox_ids = []
    for extract in data_list:
        bbox_ids.extend(extract_bbox_ids(extract))

    reductions["bbox_num_masks"] = [
        len(list(g[1])) for g in itertools.groupby(sorted(bbox_ids))
    ]
    reductions["bbox_keys"] = [g[0] for g in itertools.groupby(bbox_ids)]

    return reductions