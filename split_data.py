import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # delete the original file path if it exists
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # ensure the random select is same every time
    random.seed(0)

    # 20% move in to val
    split_rate = 0.2

    # 指向你解压后的flower_photos文件夹
    cwd = os.getcwd()
    image_pth = os.path.join(cwd, "car")
    mask_pth = os.path.join(cwd, "car_masks")
    # origin_flower_path = os.path.join(data_root, "flower_photos")
    # assert os.path.exists(origin_flower_path), "path '{}' does not exist.".format(origin_flower_path)

    # flower_class = [cla for cla in os.listdir(origin_flower_path)
    # if os.path.isdir(os.path.join(origin_flower_path, cla))]

    # create folders for train dataset
    train_root = os.path.join(cwd, "train_images")
    train_mask_root = os.path.join(cwd, "train_masks")
    mk_file(train_root)
    mk_file(train_mask_root)

    # create folders for validation dataset
    val_root = os.path.join(cwd, "val_images")
    val_mask_root = os.path.join(cwd, "val_masks")
    mk_file(val_root)
    mk_file(val_mask_root)

    images = os.listdir(image_pth)
    masks = os.listdir(mask_pth)
    num = len(images)
    # random select indexs of images
    eval_index = random.sample(images, k=int(num * split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            # copy image to validation folder
            original_path = os.path.join(image_pth, image)
            original_mask_path = os.path.join(mask_pth, image).replace(".jpg", "_mask.gif")
            copy(original_path, val_root)
            copy(original_mask_path, val_mask_root)
        else:
         # copy image to train folder
            original_path = os.path.join(image_pth, image)
            original_mask_path = os.path.join(mask_pth, image).replace(".jpg", "_mask.gif")
            copy(original_path, train_root)
            copy(original_mask_path, train_mask_root)
            print("\r[Processing [{}/{}]".format(index + 1, num), end="")  # processing bar


print()

print("processing done!")

if __name__ == '__main__':
    main()
