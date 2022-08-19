
print("ok3")


import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import FSRCNN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr

import time

# device : cpu
# fsrcnn dur time: 7.64

# pre_process dur time: 1.252
# fsrcnn dur time: 6.231
# post_process dur time: 1.473
# total fsrcnn dur time: 8.956

# device : cuda:0
# fsrcnn dur time: 2.59

# pre_process dur time: 1.116
# fsrcnn dur time: 0.010
# post_process dur time: 1.540
# total fsrcnn dur time: 2.666

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default="fsrcnn_x2.pth")
    parser.add_argument('--image-file', type=str, default="data/DN_Ref_spp32x16.bmp")
    #parser.add_argument('--image-file', type=str, default="data/butterfly_GT.bmp")
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    #device = torch.device('cpu')
    print(f"device : {device}")
    model = FSRCNN(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    bicubic = image.resize((image_width* args.scale, image_height* args.scale), resample=pil_image.BICUBIC)
    #lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    #bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    # warm-up start #########################################################################################

    hr2, _ = preprocess(image, device)
    with torch.no_grad():
        preds2 = model(hr2).clamp(0.0, 1.0)

    # warm-up end #########################################################################################

    start = time.time()

    hr, _ = preprocess(image, device)
    _, ycbcr = preprocess(bicubic, device)

    start_model = time.time()

    with torch.no_grad():
        preds = model(hr).clamp(0.0, 1.0)

    end_model = time.time()

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)

    end = time.time()

    print('pre_process dur time: {:.3f}'.format(start_model - start))
    print('fsrcnn dur time: {:.3f}'.format(end_model - start_model))
    print('post_process dur time: {:.3f}'.format(end - end_model))
    print('total fsrcnn dur time: {:.3f}'.format(end - start))

    #psnr = calc_psnr(hr, preds)
    #print('PSNR: {:.2f}'.format(psnr))

    output.save(args.image_file.replace('.', '_fsrcnn_x{}.'.format(args.scale)))
