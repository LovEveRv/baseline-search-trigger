import torch
import torch.nn as nn
import torch.optim as optim

log_interval = 10


def add_patch(img, patch):
    # images and patches are batched
    # shape: [batch_size, c, w, h]
    w = img.shape[2]
    h = img.shape[3]
    tw = patch.shape[2]
    th = patch.shape[3]
    # place at lower right
    box_leftup_x = w - tw
    box_leftup_y = h - th
    imgcopy = img.clone()
    imgcopy[:, :, box_leftup_x:box_leftup_x+tw, box_leftup_y:box_leftup_y+th] = patch[:, :, :, :]
    return imgcopy


def optimize(model, img, patch, lr, criterion, force_embed, epochs=100):
    batch_size = img.shape[0]
    optimizer = optim.Adam([patch], lr)
    for i in range(epochs):
        input_img = add_patch(img, patch)
        _, embed = model(input_img)
        loss = criterion(embed, force_embed.repeat(batch_size, 1))
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
    #     if log_interval > 0 and i % log_interval == 0:
    #         print('epoch {}: patch loss = {}'.format(i, loss.item()))
    # print()
    return patch
