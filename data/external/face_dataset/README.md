# Face Dataset Instructions

## Step 1: Download the Dataset

1. Go to the following [LINK](https://drive.google.com/drive/folders/1QwKTLP3DdnRZ8NHg_RHQcBj5rVXP0MiC "SF-TL54 dataset") to access the dataset.
2. Download the desired dataset folder.

## Step 2: Organize the Dataset

1. After downloading, extract the dataset if it is in a compressed format.
2. Move the `train`, `test`, and `val` folders into the `face_dataset` directory.

   The folder structure should look like this:

   ```

   face_dataset/

   ├── train/

   ├── test/

   └── val/

   ```

## Step 3: Data Preprocessing

1. Run the data preprocessing script to prepare the dataset for training and evaluation.
2. Ensure all dependencies for preprocessing are installed before running the script.

## Step 4: Create the YAML File

1. Create a YAML file named `sftl54.yaml` in the `config` directory if file doesn't exist.
2. Add the following content to the file:

   ```yaml
   train: ../data/processed/face_dataset/train
   val: ../data/processed/face_dataset/val

   # Keypoints
   kpt_shape: [2, 2]  # [number of keypoints, number of dim]
   flip_idx: [1, 0]

   # number of classes
   nc: 1

   # class names
   names: 
     0: face
   ```
3. Replace `path to train` and `path to val` with the actual paths to your training and validation datasets.

You're now ready to use the dataset for your project!
