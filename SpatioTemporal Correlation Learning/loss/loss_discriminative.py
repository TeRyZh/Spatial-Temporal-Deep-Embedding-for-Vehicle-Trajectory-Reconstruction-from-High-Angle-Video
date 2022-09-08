import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import random

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def discriminative_loss(embedding, labels_gt, neighbor_masks, alpha = 1, beta = 1):

    # embedding (batch_size, embed_dim, H, W)
    # labels_gt (batch_size, 1, H, W)
    # neighbor_masks(batch_size, neighbor_max, neighbor_max)  # not include background 0

    # variance: pixel embeding distance for each object variance
    # Discriminative: pair embeding distance between object instances

    batch_size = embedding.shape[0]
    embed_dim = embedding.shape[1]
    var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    diff_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    
    for b in range(batch_size):
        embedding_b = embedding[b] # (embed_dim, H, W)
        labels_gt_b = labels_gt[b] # (1, H, W)
        # print("labels_gt_b shape: ", labels_gt_b.shape)
        traj_adjacency_masks = neighbor_masks[b]

        labels = torch.unique(labels_gt_b)
        labels = labels[labels!=0]
        # print("labels: ", labels)
        num_id = len(labels)

        if num_id==0:
            # please refer to issue here: https://github.com/harryhan618/LaneNet/issues/12
            _nonsense = embedding.sum()
            _zero = torch.zeros_like(_nonsense)
            var_loss = var_loss + _nonsense * _zero
            diff_loss = diff_loss + _nonsense * _zero
            continue

        # ---------------  var_loss  ------------------
        centroid_mean = []

        for idx in labels:
            seg_mask_i = torch.squeeze(labels_gt_b == idx) 

            if not seg_mask_i.any():
                continue

            # print('seg_mask_i: ', seg_mask_i.shape)
            # print("embedding_b: ", embedding_b.shape)

            embedding_i = torch.t(embedding_b[:, seg_mask_i]) # get forground pixel positions # (num_pixel, embed_dim)

            # print("embedding_i shape: ", embedding_i.shape) 

            mean_i = torch.reshape(torch.mean(embedding_i, dim=0), (1,-1))   # (1, embed_dim)
            # print("mean_i shape: ", mean_i.shape)

            centroid_mean.append(mean_i)

            pixel_vars = 0

            vars_instance = torch.mean(1 - (1 + sim_matrix(embedding_i, mean_i))/2)  

            # print("vars_instance: " , vars_instance)
            
            var_loss = var_loss + vars_instance 

        centroid_mean = torch.squeeze(torch.stack(centroid_mean, dim = 1))  # (num_pixel, embed_dim)
        # print("centroid_mean shape: ", centroid_mean.shape)

        # ----------   differentiation_loss  -------------
        if num_id > 1:

            diff = (1 + sim_matrix(centroid_mean, centroid_mean))/2  # shape (num_id, num_id)
            diff = diff[traj_adjacency_masks[:num_id, :num_id] > 0]  # only care trajectories that share same time window

            # print("diff Loss: " , torch.mean(diff))

            # divided by two for double calculated loss above, for implementation convenience
            diff_loss = diff_loss + torch.mean(diff)


    var_loss = var_loss / batch_size
    diff_loss = diff_loss / batch_size

    Loss  = alpha*var_loss + beta*diff_loss

    return Loss


if __name__ == "__main__":

    seed = 555
    np.random.seed(seed)
    random.seed(seed)

    embedding = torch.Tensor(np.random.random((4, 16, 512, 512)).astype(np.float32)).cuda()
    labels_gt = torch.Tensor(np.random.random((4, 1, 512, 512)).astype(np.float32)).cuda()
    neighbor_masks = torch.Tensor(np.random.random((4, 100, 100)).astype(np.float32)).cuda()

    Loss = discriminative_loss(embedding, labels_gt, neighbor_masks, alpha = 1, beta = 1)
    # print("total Loss: ", Loss)
    # embedding (batch_size, embed_dim, H, W)
    # labels_gt (batch_size, embed_dim, H, W)
    # neighbor_masks(batch_size, H, W)