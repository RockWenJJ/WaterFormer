import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
from pdb import set_trace as stx
import numpy as np

# from waterformer.train import parse_options
from waterformer.utils.options import parse

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test WaterFormer on your own images.')
    parser.add_argument('--opt', type=str, required=True, help='Path to the config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint')
    parser.add_argument('--input-dir', default='./input', type=str, help='Input directory')
    parser.add_argument('--output-dir', default='./output', type=str, help='Output directory')
    parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')

    args = parser.parse_args()

    opt = parse(args.opt, is_train=False)
    checkpoint = args.checkpoint
    input_dir = args.input_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']

    if any([input_dir.endswith(ext) for ext in extensions]):
        files = [input_dir]
    else:
        files = []
        for ext in extensions:
            files.extend(glob(os.path.join(input_dir, '*.'+ext)))
        files = natsorted(files)
    
    if len(files) == 0:
        raise Exception(f'No files found at {input_dir}')

    parameters = opt['network_g']

    arch_type = parameters.pop('type')
    load_arch = run_path(f'./waterformer/models/archs/{arch_type.lower()}_arch.py')
    model = load_arch[arch_type](**parameters)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    weights = torch.load(checkpoint)
    model.load_state_dict(weights['params'], strict=False)
    model.eval()

    img_multiple_of = 8

    with torch.no_grad():
        for file_ in tqdm(files):
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
            
            img = load_img(file_)

            input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)

            # Pad the input if not_multiple_of 8
            height,width = input_.shape[2], input_.shape[3]
            H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
            padh = H-height if height%img_multiple_of!=0 else 0
            padw = W-width if width%img_multiple_of!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
            
            try:
                if args.tile is None:
                    ## Testing on the original resolution image
                    restored = model(input_)
                else:
                    # test the image tile by tile
                    b, c, h, w = input_.shape
                    tile = min(args.tile, h, w)
                    assert tile % 8 == 0, "tile size should be multiple of 8"
                    tile_overlap = args.tile_overlap

                    stride = tile - tile_overlap
                    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                    E = torch.zeros(b, c, h, w).type_as(input_)
                    W = torch.zeros_like(E)

                    for h_idx in h_idx_list:
                        for w_idx in w_idx_list:
                            in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                            out_patch = model(in_patch)
                            out_patch_mask = torch.ones_like(out_patch)

                            E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                            W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
                    restored = E.div_(W)

                restored = torch.clamp(restored, 0, 1)

                # Unpad the output
                restored = restored[:,:,:height,:width]

                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                restored = img_as_ubyte(restored[0])

                f = os.path.splitext(os.path.split(file_)[-1])[0]

                f = os.path.split(file_)[-1]

                save_img(os.path.join(output_dir, f), restored)
            except Exception as e:
                print(e)

    print(f'Output images saved at {output_dir}')