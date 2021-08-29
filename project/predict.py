"""Model predict."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 08月 27日 星期五 21:17:31 CST
# ***
# ************************************************************************************/
#
import argparse
import glob
import os
import pdb  # For debug

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from data import grid_image
from model import get_model, model_device, model_setenv

if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/Image_MultiScale_Tone.pth",
        help="checkpint file",
    )
    parser.add_argument("--input", type=str, default="images/*.png", help="input image")
    parser.add_argument("--output", type=str, default="output", help="output directory")

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model_setenv()
    model = get_model(args.checkpoint)
    device = model_device()
    model = model.to(device)
    model.eval()

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    image_filenames = glob.glob(args.input)
    progress_bar = tqdm(total=len(image_filenames))

    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        input_tensor = totensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        # .clamp(0, 1.0).squeeze()

        image = grid_image([input_tensor, output_tensor], nrow=1)

        image.save("{}/{}".format(args.output, os.path.basename(filename)))
