'''
Spectrum Mask Extractor
---------------------
This module provides a function to extract masks drawn on slitless
spectrocopic images.
'''
from collections import OrderedDict
import numpy as np
import copy
from .extractor_wrapper import extractor_wrapper

@extractor_wrapper()
def spectrum_mask_extractor(classification, gold_standard=False, **kwargs):
    '''Extract annotations from a line tool with a text sub-task

    Parameters
    ----------
    classification : dict
        A dictionary containing an `annotations` key that is a list of
        panoptes annotations

    Returns
    -------
    extraction : dict
        A dictionary with one key for each `bounding box`. 
        TODO: Add description of the structure of this dictionary.
    '''
    blank_frame = OrderedDict(
        [
            ("bbox_id", []),
            ("sam_points", {"x": [], "y": [], "label": [], "order": []}),
            ("latest_sam_mask", []),
            ("composite_mask", []),
            ("sam_model", []),
        ]
    )
    extract = OrderedDict()
    if len(classification['annotations']) > 0:
        # there is only one task so we can just take the first annotation
        annotation = classification['annotations'][0]
        # there is one "value" per bounding box
        for value in annotation['value']:
            frame = 'frame{0}'.format(value.get('frame', 0))
            extract.setdefault(frame, copy.deepcopy(blank_frame))
            # annotation id
            extract[frame]['bbox_id'].append(value['annotationId'])
            # SAM points
            extract[frame]["sam_points"]["x"].append([sp["x"] for sp in value["samPoints"]])
            extract[frame]["sam_points"]["y"].append([sp["y"] for sp in value["samPoints"]])
            extract[frame]["sam_points"]["label"].append([sp["label"] for sp in value["samPoints"]])
            extract[frame]["sam_points"]["order"].append([sp["pointId"] for sp in value["samPoints"]])

            # SAM models
            extract[frame]['sam_model'].append(value['samModelId'])
            # SAM masks
            extract[frame]['latest_sam_mask'].append(value['latestSamMask'])
            # composite masks
            extract[frame]['composite_mask'].append(value['compositeMask'])
    return extract