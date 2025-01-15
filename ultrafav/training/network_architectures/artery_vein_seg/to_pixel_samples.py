import torch
import torch.nn.functional as F


def n8_encode(img):
    B, H, W = img.shape
    directions = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]],
                              device=img.device)

    # create a padded image, which is used to get the n8 maps
    padded_img = F.pad(img.unsqueeze(1), (1, 1, 1, 1), mode='constant', value=-1).squeeze(1)
    n8_maps = torch.zeros((B, H * W, 8), device=img.device, dtype=img.dtype)

    # obtain n8 maps
    for k, (dx, dy) in enumerate(directions):
        shifted_img = padded_img[:, 1 + dx:1 + dx + H, 1 + dy:1 + dy + W]
        same_value = shifted_img == img
        n8_maps[:, :, k] = same_value.view(B, -1)

    return n8_maps  # .permute(0, 2, 1)


def to_n8_maps(img):
    if isinstance(img, list):
        n8_maps = []
        for i in img:
            if len(i.shape) == 4 and i.shape[1] == 1:
                i = i.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
            n8_maps.append(n8_encode(i))
        return n8_maps
    else:
        if len(img.shape) == 4 and img.shape[1] == 1:
            img = img.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
        return n8_encode(img)


if __name__ == '__main__':
    # define a tensor contains 0 and 1 and 2, size=[batch size, 1, 5,5]
    img = torch.tensor([[[[1, 1, 2, 0, 0],
                          [0, 1, 2, 0, 0],
                          [2, 2, 2, 2, 0],
                          [1, 0, 2, 2, 0],
                          [2, 0, 0, 2, 0]]]])
    n8_maps = to_n8_maps(img)
    print(n8_maps)
