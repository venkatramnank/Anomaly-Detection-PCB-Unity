# Anomaly Detection For PCB (Code documetation)

# Introduction

This is an implementation of the paper [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/pdf/2011.08785).

This code is heavily borrowed from the unofficial implementation by [xiahaifeng1995](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master) and the efficient net version implementation by [yougjae](https://github.com/youngjae-avikus/PaDiM-EfficientNet/tree/master).

The final code implements EfficientNet with PaDiM with slight changes in terms of the original implementation. The code also (in future) extends the model into a pipeline that takes in frames from a real time camera (HKVision) and produces inference.

---

# Setup

## Requirements

- python >= 3.7
- pytorch >= 1.5
- torchvision
- pillow==9.5.0
- tqdm
- sklearn
- matplotlib
- scikit-image
- [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

## Datasets

For initial testing , the MVTec AD datasets : Download from [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad/) is used.

Custom PCB data created using Unity. The tree structure required for running training:

```
pcb
├── ground_truth
│   ├── misplaced_objects
│   │   ├── step0.camera_solo_1_mask.png
│   │   ├── step0.camera_solo_2_mask.png
│   │   ├── step0.camera_solo_3_mask.png
│   │   ├── step0.camera_solo_4_mask.png
│   │   ├── step0.camera_solo_5_mask.png
│   │   ├── step0.camera_solo_6_mask.png
│   │   ├── step0.camera_solo_7_mask.png
│   │   └── step0.camera_solo_mask.png
│   └── missing_objects
│       ├── step0.camera_solo_1_mask.png
│       ├── step0.camera_solo_2_mask.png
│       ├── step0.camera_solo_3_mask.png
│       └── step0.camera_solo_mask.png
├── test
│   ├── good
│   │   ├── step0.camera_solo.png
│   │   ├── step0.camera_solo_1.png
│   │   ├── step1.camera_solo.png
│   │   └── step1.camera_solo_1.png
│   ├── misplaced_objects
│   │   ├── step0.camera_solo.png
│   │   ├── step0.camera_solo_1.png
│   │   ├── step0.camera_solo_2.png
│   │   .
    │   .
│   └── missing_objects
│       ├── step0.camera_solo.png
│       ├── step0.camera_solo_1.png
│       .
        .
└── train
    └── good
        ├── step0.camera_solo.png
        ├── step0.camera_solo_1.png
        ├── step0.camera_solo_3.png
        ├── step0.camera_solo_4.png
        .
        .

toolbox
├── ground_truth
│   ├── misplaced_objects
│   │   ├── step0.camera_solo_1_mask.png
│   │   ├── step0.camera_solo_2_mask.png
│   │   ├── step0.camera_solo_3_mask.png
│   │   ├── step0.camera_solo_4_mask.png
│   │   ├── step0.camera_solo_5_mask.png
│   │   ├── step0.camera_solo_6_mask.png
│   │   ├── step0.camera_solo_7_mask.png
│   │   └── step0.camera_solo_mask.png
│   └── missing_objects
│       ├── step0.camera_solo_1_mask.png
│       ├── step0.camera_solo_2_mask.png
│       ├── step0.camera_solo_3_mask.png
│       └── step0.camera_solo_mask.png
├── test
│   ├── good
│   │   ├── step0.camera_solo.png
│   │   ├── step0.camera_solo_1.png
│   │   ├── step1.camera_solo.png
│   │   └── step1.camera_solo_1.png
│   ├── misplaced_objects
│   │   ├── step0.camera_solo.png
│   │   ├── step0.camera_solo_1.png
│   │   ├── step0.camera_solo_2.png
│   │   .
    │   .
│   └── missing_objects
│       ├── step0.camera_solo.png
│       ├── step0.camera_solo_1.png
│       .
        .
└── train
    └── good
        ├── step0.camera_solo.png
        ├── step0.camera_solo_1.png
        ├── step0.camera_solo_3.png
        ├── step0.camera_solo_4.png
        .
        .

```

It must be noted you will need to change the list of class names in `datasets/mvtec.py` in line 13 as:

```
CLASS_NAMES = ['pcb', 'toolbox']
```

## Training

In order to train the Efficient net model

```
 python efficient_main.py -d [dataset location] -s [results storage directory] --training
```

## Inference

Create a folder inside the directory called model_pkl_efficientnet-b4. The b4 here refers to the model architecture used, thus needs to be replaced of the following format

```
model_pkl_<model name>

```

Place the trained model inside the folder. The model trianed for the PCB can be found [here](https://drive.google.com/file/d/1h28jrUBAWC0qK6xuzVMcmFyPKlIyZUKM/view?usp=sharing).

One can also use gdown to download the pkl file.

```
$ pip install gdown
$ gdown --fuzzy "link to the google drive file"

```

Then run the inference code :

```
python inference_test.py -t data_path_of_single_test_image -s save_path_where_the_pkl_folder_exists_and_where_results_are_stored
```

---

---

# Code Documentation

### datasets/mvtec.py

This script builds the  **`MVTecDataset`** class, which is inspired by the famous MVTec Anomaly Detection dataset. This **************dataset************** class is used to load the data into the Pytorch ********************dataloader.********************

This **`MVTecDataset`** class is designed to provide a convenient way to load, preprocess, and access data instances from the MVTEC Anomaly Detection dataset, which is essential for training and evaluating anomaly detection models.

The below line needs to be updated with the class names to be a part of the dataset built by the MVTecDataset class.

```jsx
CLASS_NAMES = ['pcb']
```

Let's break down the key components and functionalities of this class:

1. **Constructor (`__init__`)**:
    - The constructor initializes the dataset object with various parameters:
        - **`dataset_path`**: The root path where the MVTEC dataset is located.
        - **`class_name`**: The name of the object class within the dataset that will be used (e.g., 'bottle').
        - **`is_train`**: A boolean indicating whether the dataset is for training (**`True`**) or testing (**`False`**).
        - **`resize`**: The target size to which images will be resized.
        - **`cropsize`**: The size of images after center cropping.
    - The constructor asserts that the provided **`class_name`** is valid based on a predefined list of class names.
    - It sets up instance variables to store dataset-related information such as paths, class name, and transformation functions.
2. **Data Loading (`load_dataset_folder`)**:
    - This method loads the image and ground truth mask file paths for the specified class and dataset split (train or test).
    - It iterates through different types of images (e.g., 'good', 'defective'), collecting paths and labels accordingly.
    - For 'good' images, **`y`** (label) is set to 0, and the **`mask`** is set to **`None`**.
    - For 'defective' images, **`y`** is set to 1, and the **`mask`** is the corresponding ground truth mask image path.
3. **Data Transformation (`__getitem__`)**:
    - This method retrieves a single data instance (image, label, and mask) based on the provided index (**`idx`**).
    - It opens the image using PIL and applies transformations to preprocess it:
        - Resizing using bilinear interpolation.
        - Center cropping to the specified **`cropsize`**.
        - Converting the image to a tensor.
        - Normalizing pixel values using predefined mean and standard deviation values.
    - If the label **`y`** is 0 (indicating a 'good' image), the mask is created as an all-zero tensor of appropriate size.
    - If the label **`y`** is 1 (indicating a 'defective' image), the corresponding mask image is loaded using PIL and then transformed similarly to the image.
4. **Length (`__len__`)**:
    - This method returns the total number of data instances in the dataset, which corresponds to the length of the **`x`** list (image paths).
5. **Usage**:
    - You can instantiate an object of the **`MVTecDataset`** class by providing appropriate parameters.
    - To access a data instance, you can use the indexing operator (**`dataset[idx]`**), where **`idx`** ranges from 0 to **`len(dataset) - 1`**.
6. **Transforms**:
    - The class defines two sets of transformations: **`transform_x`** for image preprocessing and **`transform_mask`** for mask preprocessing.
    - These transformations are applied to images and masks, respectively, during the data loading process.
7. **Notes**:
    - The dataset is divided into 'good' and 'defective' image types, where 'good' images have a label of 0, and 'defective' images have a label of 1.
    - The **`mask`** corresponds to the ground truth mask image that indicates anomalies in 'defective' images.

---

### Utils Folder

***good_data_extractor.py***

This Python script is designed to create a subset of "good" data from a source directory containing images, similar to the structure of the MVTec dataset. This subset of data is intended for use with the PadiM and PatchCore anomaly detection systems. The script recursively traverses through the source directory, identifies image files with the ".png" extension, and then copies these images to a designated destination directory.

Let's break down the script step by step:

1. **Import Statements**:
    - **`os`**: Provides functions to interact with the operating system.
    - **`numpy as np`**: NumPy library for numerical operations.
    - **`shutil`**: Offers high-level file operations like copying.
    - **`pathlib.Path`**: A module to work with paths in a more object-oriented way.
2. **Source and Destination Directories**:
    - **`src_dir`**: The source directory from which images will be extracted. This is the directory that contains the images to be processed.
    - **`dest_dir`**: The destination directory where the "good" images will be copied. This is where the "good" data similar to the MVTec dataset will be created.
3. **List of Image Files**:
    - **`image_files`**: An empty list that will store the paths of identified image files.
4. **Loop Through Source Directory**:
    - The script uses the **`os.walk`** function to traverse through the source directory and its subdirectories.
    - For each directory being traversed (**`root`**), it retrieves the list of files in that directory (**`files`**).
    - It then checks if each file has a ".png" extension using the **`.endswith('.png')`** method.
    - If the file is an image (based on the extension), its full path is appended to the **`image_files`** list.
5. **Copy Image Files**:
    - The script iterates through the list of image files.
    - For each image file, it constructs the destination path by:
        - Creating the directory structure in the destination directory using **`os.path.relpath`** to maintain the directory hierarchy.
        - Combining the **`dest_dir`** with the relative path from the source directory to the image file.
    - It creates the necessary directories in the destination path using **`os.makedirs`** with **`exist_ok=True`** to avoid errors if the directory already exists.
    - It then copies the image file from the source path to the destination path using **`shutil.copy2`**, which preserves metadata like creation and modification times.
6. **Renaming Files (Note)**:
    - The script provides a note to use a Windows PowerShell command to rename files. This command renames the files by appending "_solo_16" to the original filename, just before the extension.
7. **Execution**:
    - To use the script, replace the **`src_dir`** and **`dest_dir`** paths with the appropriate paths on your system.
    - The script then recursively searches for image files in the source directory and copies them to the destination directory, maintaining the directory structure.
    

***mask_builder.py***

This Python script is designed to build binary masks from segmentation masks using RGB color information. These masks are useful for isolating specific regions of interest within an image. The script is specifically written for a use case involving Unity-generated JSON files that provide label information for different regions in the segmentation mask. However, it's noted that the script requires manual configuration of the RGB values and it's marked as a TODO to automate this process.

Here's a breakdown of the script:

1. **Import Statements**:
    - **`numpy as np`**: NumPy library for numerical operations.
    - **`cv2`**: OpenCV library for computer vision tasks.
    - **`matplotlib.pyplot as plt`**: Matplotlib library for plotting (not used in the script, possibly for debugging).
2. **`segmentation_mask_convertor_nail` Function**:
    - This function takes an image path (**`img_path`**) as input and converts a specific label in the segmentation mask to a binary mask.
    - It reads the image from the given path using OpenCV, converting the colors from BGR to RGB.
    - It defines two RGB values (**`rgb_val`** and **`rgb_val2`**) that correspond to specific labels in the JSON file.
    - It then creates lower and upper color range values based on the provided RGB values.
    - Using these ranges, the script applies **`cv2.inRange`** to create masks for each RGB value.
    - A combined mask is created by performing a bitwise OR operation between the individual masks.
    - A blank result image is initialized.
    - The result mask is generated by setting pixels in the result image to white (255, 255, 255) where the corresponding pixels in the individual masks are white.
    - The resulting mask is saved as an image using **`cv2.imwrite`**.
3. **Image Path and Function Call**:
    - The script provides a test image path (**`test_img_path`**) to demonstrate the function.
    - The **`segmentation_mask_convertor_nail`** function is then called with the test image path as an argument.
4. **Execution and Output**:
    - To use the script, replace the **`test_img_path`** with the path of the segmentation mask image you want to process.
    - The script processes the image, extracts the specified regions based on RGB values, and generates a binary mask.
    - The resulting mask is saved as an image in the provided path.
5. **Notes**:
    - The script currently requires manual configuration of the RGB values (**`rgb_val`** and **`rgb_val2`**) based on the JSON label information. It's noted as a TODO to automate this process, which would likely involve reading the label information from the JSON files.

---

## efficient_modified.py

This code defines a modified version of the EfficientNet architecture called **`EfficientNetModified`**. This modification enables the extraction of intermediate features from specific blocks of the EfficientNet model, as opposed to just the last layer's features. This modified version inherits from the original **`EfficientNet`** class, and you can use it to extract features from different blocks within the EfficientNet architecture.

Here's an explanation of the key components and functions in this code:

1. **`EfficientNetModified` Class**:
    - This class extends the original **`EfficientNet`** class from the **`efficientnet_pytorch`** library.
    - The purpose of this modification is to allow extraction of features from specific blocks of the EfficientNet model.
2. **`extract_features` Method**:
    - This method takes two parameters: **`inputs`** (the input tensor) and **`block_num`** (a list of block numbers from which to extract features).
    - It initializes an empty list **`feat_list`** to store the extracted features.
    - The method starts with the stem operation on the input tensor (**`_conv_stem`** followed by batch normalization and Swish activation).
    - It iterates through the blocks of the EfficientNet model and applies the blocks to the input tensor.
    - During each iteration, the method checks if the current iteration count (**`iter`**) matches any of the block numbers specified in **`block_num`**.
    - If the iteration count matches, the current feature tensor is appended to the **`feat_list`**.
    - The method concludes with the head operation (**`_conv_head`** followed by batch normalization and Swish activation).
    - Finally, the list of extracted features (**`feat_list`**) is returned.
3. **`extract_entire_features` Method**:
    - This method is similar to the **`extract_features`** method but captures features from all blocks, not just specific ones.
    - It initializes the list of extracted features with the stem's output.
    - It then iterates through all the blocks, appending the output of each block to the list of extracted features.
    - The method concludes with the head operation, and the final feature tensor from the head is appended to the list.
    - This method is useful if you want to capture features from every block without specifying specific block numbers.

---

## efficient_main.py

This script implements a process for training and evaluating the PaDiM (Patch Distribution Modeling) anomaly detection model using the EfficientNet architecture. Let's break down the script into sections and explain its functionality:

### embedding_concat

1. The function takes two input tensors, **`x`** and **`y`**, representing feature maps from different layers of the neural network.
2. It extracts the dimensions of these tensors: **`B`** (batch size), **`C1`** (number of channels for **`x`**), **`H1`** (height of **`x`**), and **`W1`** (width of **`x`**). Similarly, it extracts the dimensions for **`y`** as **`C2`**, **`H2`**, and **`W2`**.
3. The variable **`s`** is calculated as the downsampling factor between the two layers. It's used to determine how much to downsample **`x`** to match the dimensions of **`y`**.
4. The function then applies a series of operations to concatenate the embeddings:
    - It unfolds the **`x`** tensor using a specified kernel size and stride, effectively creating overlapping patches of size **`s`** on **`x`**.
    - The unfolded **`x`** tensor is reshaped to have dimensions **`(B, C1, -1, H2, W2)`**. The **`1`** accounts for the number of patches created during the unfolding process.
    - A new tensor **`z`** is initialized with zeros, having dimensions **`(B, C1 + C2, num_patches, H2, W2)`**. This tensor will store the concatenated embeddings.
    - A loop iterates over the patches in **`x`**, and for each patch, concatenates the corresponding patch from **`y`** along the channel dimension. This effectively concatenates the embeddings from both layers for each patch.
5. After all patches are processed, the tensor **`z`** is reshaped to have dimensions **`(B, -1, H2 * W2)`**.
6. Finally, the **`F.fold`** function is applied to upsample the tensor **`z`** back to the original dimensions of **`x`** (**`H1`** and **`W1`**) using a kernel size of **`s`** and stride **`s`**.

The resulting tensor is a concatenated representation of the embeddings from both **`x`** and **`y`**, with proper spatial alignment. This concatenated tensor can then be used for further processing in the model.

### calc_covinv

Here's what the code does:

1. The function takes four arguments:
    - **`embedding_vectors`**: A tensor containing concatenated embedding vectors. The tensor shape is assumed to be **`(batch_size, num_channels, H, W)`** where **`batch_size`** is the number of samples in the batch, **`num_channels`** is the number of channels in each embedding, and **`H`** and **`W`** are the height and width of the spatial dimensions.
    - **`H`**: The height of the spatial dimensions.
    - **`W`**: The width of the spatial dimensions.
    - **`C`**: The number of channels in the embeddings.
2. The function iterates over each spatial position **`(i, j)`** in the tensor, where **`i`** ranges from **`0`** to **`H - 1`** and **`j`** ranges from **`0`** to **`W - 1`**. This is achieved using the loop **`for i in range(H * W):`**.
3. For each spatial position **`(i, j)`**, the function calculates the inverse covariance matrix using the **`np.cov`** function. It extracts the values for the **`i`**th and **`j`**th positions across all samples and channels. **`rowvar=False`** ensures that each row represents a variable and each column represents an observation.
4. A small regularization term is added to the covariance matrix using **`0.01 * np.identity(C)`**. This regularization helps prevent issues when the covariance matrix is close to singular.
5. The function calculates the inverse of the covariance matrix using **`np.linalg.inv`**.
6. The calculated inverse covariance matrix is yielded back to the caller using the **`yield`** statement. The function operates as a generator, allowing the caller to retrieve the inverse covariance matrices one by one without needing to store them all in memory.

### main function

The code encompasses various tasks including data loading, model evaluation, plotting of ROC curves, and saving results. Let's break down the key components and functionalities of this function:

1. **Creating Directories**:
    - The function starts by creating a directory to save data related to the model's training.
2. **Preparing ROCAUC Score Plotting**:
    - Initializes variables for ROCAUC scores and prepares a subplot for plotting ROC curves.
3. **Model Configuration**:
    - Based on the provided **`args.arch`**, specific configurations are set for different architectures ('b0', 'b1', 'b4', 'b7') of the EfficientNet model.
    - This includes specifying block numbers and the number of filters in the model's architecture.
4. **Creating Random Seed**:
    - Calls the **`create_seed`** function to set a random seed, possibly for reproducibility purposes.
5. **Model Configuration and Initialization**:
    - The EfficientNet model is moved to the specified device (likely GPU) using **`eff_model.to(device)`**.
6. **Data Loading and Processing**:
    - For each class in the MVTec dataset, training and test datasets are loaded using the **`MVTecDataset`** class and corresponding data loaders are created.
7. **Feature Extraction (Training)**:
    - If the **`args.training`** flag is set to **`True`**, the code enters a block for feature extraction during the training phase.
    - The EfficientNet model is set to evaluation mode using **`eff_model.eval()`**.
    - For each batch of training data, the features are extracted for different layers of the model.
    - These features are then concatenated using the **`embedding_concat`** function.
    - The mean and inverse covariance matrices are calculated for the concatenated features.
    - These calculated values are saved to a file for later use.
8. **Feature Loading (Testing)**:
    - If **`args.training`** is **`False`**, the code loads the previously calculated mean and inverse covariance matrices from the file.
9. **Feature Extraction (Testing)**:
    - For each batch of test data, features are extracted for different layers of the model.
    - These features are concatenated using the **`embedding_concat`** function.
    - The Mahalanobis distance is calculated for each pixel based on the extracted features.
10. **Scoring and Thresholding**:
    - The Mahalanobis distances are normalized to obtain a score map.
    - Gaussian filtering is applied to the score map.
    - Image-level ROC AUC scores are calculated based on the maximum score for each image.
    - Optimal thresholds for image-level anomaly detection and pixel-level anomaly detection are determined.
11. **Pixel-level Anomaly Detection**:
    - Precision-recall curves are calculated for pixel-level anomaly detection.
    - The F1 score is computed for different thresholds, and the optimal threshold is selected.
12. **ROC Curve Plotting and Results Saving**:
    - ROC curves are plotted and saved for both image-level and pixel-level anomaly detection.
    - The results are saved to text files, including class names, ROC AUC scores, and inference times.
13. **Final Results and Plotting**:
    - The average image-level and pixel-level ROC AUC scores are calculated and printed.
    - The subplot containing ROC curves is fine-tuned and saved as a figure.
    

---

> **Note :**  The code in **********************************inference_test.py********************************** is built based on **********************************efficient_main.py,********************************** modified in such a way that it gives the inference only for one image
> 

---
