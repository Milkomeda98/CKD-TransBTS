import os
import shutil
import argparse

def merge_directories(base_dir, new_dir):
    # Create the new directory if it doesn't exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # List of subdirectories to merge
    sub_dirs = ['train', 'test', 'val']

    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(base_dir, sub_dir)
        
        # Iterate over each patient folder in the current subdirectory
        for patient_folder in os.listdir(sub_dir_path):
            patient_folder_path = os.path.join(sub_dir_path, patient_folder)
            
            # Make sure it's a directory
            if os.path.isdir(patient_folder_path):
                # Define the new path for the patient folder
                new_patient_folder_path = os.path.join(new_dir, patient_folder)
                
                # Move the patient folder to the new directory
                shutil.move(patient_folder_path, new_patient_folder_path)

    print("Merging completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move patient directories from train, test, and val into one directory.")
    parser.add_argument('base_dir', type=str, help="Base directory containing train, test, and val directories.")
    parser.add_argument('new_dir', type=str, help="New directory to store merged patient directories.")
    
    args = parser.parse_args()
    
    merge_directories(args.base_dir, args.new_dir)
