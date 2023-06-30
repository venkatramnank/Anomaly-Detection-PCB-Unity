"""
Python script for building good data similar to MVTec dataset for PadiM and PatchCore
Given data from solo form provided by Unity
This script essentially extracts all images recursively and copies them to a different folder.
"""

import os
import numpy as np
import shutil
from pathlib import Path

# Please replace this directory information
src_dir = r'C:\Users\kalyanak\Desktop\FiabFlexforce\AnomalyDetection\nokia_pcb_dataset\OneDrive_2023-06-29\PCB DATA\PCB IMAGES'
dest_dir = r'C:\Users\kalyanak\Desktop\FiabFlexforce\AnomalyDetection\nokia_pcb_dataset\OneDrive_2023-06-29\PCB DATA\good'

image_files = []
for root, _, files in os.walk(src_dir):
    for file in files:
        if file.lower().endswith(('.png')):
            image_files.append(os.path.join(root, file))

print(len(image_files))

for image_file in image_files:
    destination_path = os.path.join(dest_dir, os.path.relpath(image_file, src_dir))
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copy2(image_file, destination_path)

"""
NOTE: Please use the below command in windows powershell to rename files easily
Dir | Rename-Item -NewName { $_.basename + "_solo_16" + $_.extension}

"""
