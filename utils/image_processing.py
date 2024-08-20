import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt


def show_from_tensor(tensor):
    img = tensor.clone()
    img = img.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))

    plt.figure(figsize=(10, 7))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def norm_data(data):
    return (data.clip(-1, 1) + 1) / 2


def create_crops(img, num_crops=32, size1=224, aug_transform=None, noise_factor=0.22):
    p = size1 // 2
    img = torch.nn.functional.pad(img, (p, p, p, p), mode='constant', value=0)

    if aug_transform:
        img = aug_transform(img)

    crop_set = []
    for _ in range(num_crops):
        gap1 = int(torch.normal(1.2, .3, ()).clip(.43, 1.9) * size1)
        offsetx = torch.randint(0, int(size1 * 2 - gap1), ())
        offsety = torch.randint(0, int(size1 * 2 - gap1), ())

        crop = img[:, :, offsetx:offsetx + gap1, offsety:offsety + gap1]

        crop = torch.nn.functional.interpolate(crop, (224, 224), mode='bilinear', align_corners=True)
        crop_set.append(crop)

    img_crops = torch.cat(crop_set, 0)

    randnormal = torch.randn_like(img_crops, requires_grad=False)
    randstotal = torch.rand((img_crops.shape[0], 1, 1, 1)).to(img_crops.device)

    img_crops = img_crops + noise_factor * randstotal * randnormal

    return img_crops
