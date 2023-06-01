import argparse
import numpy as np
from PIL import Image


def get_images(config, index, load_dir, save_dir):
    if index%10 < 6:
        split = "train"
    elif index%10 < 8:
        split = "val"
    else:
        split = "test"
    data = np.load("{}/{}/RAVEN_{}_{}.npz".format(load_dir,config,index,split))
    images = data["image"]
    target = data["target"]
    for j in range(8):
        image = Image.fromarray(images[j])
        image.save("{}/{}.png".format(save_dir,j))
    image = Image.fromarray(images[8+target])
    image.save("{}/8.png".format(save_dir))
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--index", type=int)
    parser.add_argument("--load_dir")
    parser.add_argument("--save_dir")
    args = parser.parse_args()
    get_images(args.config, args.index, args.load_dir, args.save_dir)
    return


if __name__ == "__main__":
    main()