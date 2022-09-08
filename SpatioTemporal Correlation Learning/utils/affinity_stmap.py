import numpy as np
from scipy.ndimage import shift

def gen_offsets(shift, neighbor=8):
    assert neighbor == 4 or neighbor == 8, 'neigbor must be 4 or 8!'
    if neighbor == 4:
        return [[-shift, 0], [0, -shift]]
    else:
        return [[-shift, 0], [0, -shift], [-shift, -shift], [-shift, shift]]

def multi_offset(shifts, neighbor=8):
    out = []
    for shift in shifts:
        out += gen_offsets(shift, neighbor=neighbor)
    return out


def gen_affs_stmap(labels, offsets=[[-1,0],[0,-1]], max_neighbor = 60, ignore=False, padding=False):
    
    n_channels = len(offsets)
    
    affinities = np.zeros((n_channels,) + labels.shape, dtype=np.float32)

    masks = np.zeros((n_channels,) + labels.shape, dtype=np.uint8)

    for cid, off in enumerate(offsets): 
        shift_off = [-x for x in off] 
        shifted = shift(labels, shift_off, order=0, prefilter=False) 
        mask = np.ones_like(labels) 
        mask = shift(mask, shift_off, order=0, prefilter=False) 
        dif = labels - shifted 
        out = dif.copy() 
        out[dif == 0] = 1 
        out[dif != 0] = 0 
        if ignore: 
            out[labels == 0] = 0 
            out[shifted == 0] = 0 
        if padding: 
            out[mask==0] = 1 
        else: 
            out[mask==0] = 0 

        affinities[cid] = out 
        masks[cid] = mask 
    
    ################ find instances share same time window   ################
    # neighbor_masks is used to push different trajectories in the same window far apart

    obj_ids = np.unique(labels)

    num_instances = len(obj_ids) - 1 # not include background 0

    neighbor_masks = np.zeros((max_neighbor, max_neighbor), dtype=int)

    for obj_id in obj_ids:

        if obj_id == 0: continue

        if obj_id > np.ceil(num_instances/2):
             break

        rows_x, cols_y = np.where(labels == obj_id)

        min_col_y, max_col_y = np.min(cols_y), np.max(cols_y)

        time_win_labels = labels[:, min_col_y : max_col_y] 

        neighbor_instance_ids = np.unique(time_win_labels)
        
        for neighbor_instance_id in neighbor_instance_ids:

            if neighbor_instance_id != obj_id and neighbor_instance_id != 0:

                neighbor_masks[obj_id - 1, neighbor_instance_id - 1] = 1

    neighbor_masks = neighbor_masks + np.transpose(neighbor_masks)

    ################ find instances share same time window   ################

    return affinities, masks, neighbor_masks
