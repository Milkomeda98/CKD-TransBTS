
import os
def get_brats_folder(dataset_folder, mode):
    assert mode in ["train","train_val", "test"]
    if mode == "train":
        return os.path.join(dataset_folder, "brats2021", "train")
    elif mode == "train_val":
        return os.path.join(dataset_folder, "brats2021", "val")
    elif mode == "test" :
        return os.path.join(dataset_folder, "brats2021", "test")
