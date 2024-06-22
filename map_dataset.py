"""
map BraTS 2023 dataset with BraTS 2021 dataset
since both the dataset are same but they have different names for each patients directory
By using mapping file we can match BraTS2023 dataset with BraTS 2021 dataset for training and test


python map_dataset.py -h
"""

import pandas as pd
import os
import argparse
import shutil

# read command line args
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default= "", type= str, help= "path to the brats23 dataset")
    parser.add_argument("--dataset_selector", default= "", type= str, help= "path to the brats23 folder selector dir for each split type i.e txt files for train, test, and validation")
    parser.add_argument("--split", type = str, default="train", help= "name of the dataset split i.e. trian, test, or validaiton")
    parser.add_argument("--mapping_file", type = str, help= "path to the brats23 and brats21 mapping excel file given in the brats23 dataset")
    parser.add_argument("--destination_dir", type = str, help= "path to the destination dir")
    return parser.parse_args()

# create correspondence between brats23 and brats21
def create_brats23_brats21_mapping(file_name):
    df = pd.read_excel(file_name)
    correspondence = {}
    brats23_col = 0
    brats21_col = 1
    for ind in range(len(df)-4):
        brats23_value = df.iloc[ind, brats23_col]
        brats21_value = df.iloc[ind, brats21_col]
        correspondence[brats21_value] = brats23_value
    return correspondence

if __name__ == "__main__":
   args = read_args()
   
   # create correspondence dict
   correspondence_dict = create_brats23_brats21_mapping(args.mapping_file)

   # fetch 
   if args.dataset_selector:
       split_name = args.split
       
       # create a folder inside the destination dataset
       os.makedirs(os.path.join(args.destination_dir, split_name), exist_ok=True)

       # get split txt file
       destination_folder = os.path.join(args.destination_dir, split_name)
       split_file = os.path.join(args.dataset_selector, split_name + "_list.txt")

       with open(split_file, mode = 'r') as file:
             folders_to_copy = file.read().splitlines()
       # map the dataset
       folders_to_copy = [correspondence_dict[folder] for folder in folders_to_copy] 

       for folder_name in folders_to_copy:
            folder_name = folder_name[:19]
            source_path = os.path.join(args.dataset, folder_name)
            target_path = os.path.join(destination_folder, folder_name)

            shutil.copytree(source_path, target_path)
       print()
       print('copy complete!')

           
