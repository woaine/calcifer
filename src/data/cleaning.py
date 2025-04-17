import pandas as pd
import os
import xml.etree.ElementTree as ET
import cv2

from shutil import copyfile
from itertools import product

def clean_data(df:pd.DataFrame, data_group:str=None):
    def clean_data(df: pd.DataFrame):
        """
        Cleans and processes a given DataFrame by performing several data transformation steps.
        This function assumes the input DataFrame has specific columns related to temperature data.
        It removes unnecessary rows, sets appropriate headers, converts data types, calculates
        averages, and renames and reorders columns for further analysis.
        Args:
            df (pd.DataFrame): The input DataFrame containing raw data to be cleaned. It is expected
                to have columns named 'T_FHBC1', 'T_FHBC2', 'T_FHBC3', 'T_FHBC4', 'aveOralM', and 'T_atm'.
        Returns:
            pd.DataFrame: A cleaned and processed DataFrame with the following columns:
                - 'Tg': The average of 'T_FHBC1', 'T_FHBC2', 'T_FHBC3', and 'T_FHBC4'.
                - 'Ta': The renamed column for 'T_atm'.
                - 'Tc': The renamed column for 'aveOralM'.
        Raises:
            KeyError: If the required columns are not present in the input DataFrame.
            ValueError: If the data in the required columns cannot be converted to float.
        Notes:
            - The function removes the first row of the DataFrame and uses the second row as headers.
            - The input DataFrame is expected to have numeric data in the specified columns for
              type conversion and calculations to succeed.
        """

    # Remove the first row_data
    df = df.iloc[1:].reset_index(drop=True)
    # Make first row_data as headers
    df.columns = df.iloc[0]
    # Reset index
    df = df[1:].reset_index(drop=True)

    # Change column types to float64
    df[['T_FHBC1', 'T_FHBC2', 'T_FHBC3', 'T_FHBC4', 'aveOralM', 'T_atm']] = df[['T_FHBC1', 'T_FHBC2', 'T_FHBC3', 'T_FHBC4', 'aveOralM', 'T_atm']].astype('float64')

    # Calculate the average of T_FHBC1 to T_FHBC4
    df['Tg'] = df[['T_FHBC1', 'T_FHBC2', 'T_FHBC3', 'T_FHBC4']].mean(axis=1)

    # Keep specified columns
    df = df[['Tg', 'T_atm', 'aveOralM']]

    # Rename columns
    df = df.rename(columns={'T_atm': 'Ta', 'aveOralM': 'Tc'})

    # Reorder columns
    return df[['Tg', 'Ta', 'Tc']]

def save_clean_data_to_csv(clean_data: pd.DataFrame, path: str):
    clean_data.to_csv(path, index=False)

def make_face_dataset_dir(target_path: str = '../data/processed/face_dataset'):
    """
    Creates a directory structure for a face dataset, organizing it into 
    subdirectories for training, validation, and testing sets, each containing 
    separate folders for images and labels.
    Args:
        target_path (str): The root directory where the dataset structure 
            will be created. Defaults to '../data/processed/face_dataset'.
    The resulting directory structure will look like this:
        target_path/
            ├── train/
            │   ├── images/
            │   └── labels/
            ├── val/
            │   ├── images/
            │   └── labels/
            └── test/
                ├── images/
                └── labels/
    If the directories already exist, they will not be recreated, and no error 
    will be raised.
    """

    for (set, data) in product(['train', 'val', 'test'], ['images', 'labels']):
        os.makedirs(os.path.join(target_path, set, data), exist_ok = True)

def XMLtoTXT(source_path: str = '../data/external/face_dataset', filename: str = 'facial_data', target_path: str = '../data/processed/face_dataset', move_images: bool = True):
    """
    Converts facial annotation data from XML format to YOLO-compatible TXT format and optionally moves associated images.
    This function processes XML files containing facial annotation data, extracts bounding box and keypoint information, 
    and converts it into a format compatible with YOLO object detection models. The converted data is saved as TXT files 
    in the specified target directory. Additionally, the function can move the associated image files to the target directory.
    Args:
        source_path (str): The path to the source directory containing the XML files and images. 
                           Default is '../data/external/face_dataset'.
        filename (str): The base name of the XML files to process (without the set-specific prefix). 
                        Default is 'facial_data'.
        target_path (str): The path to the target directory where the processed TXT files and images will be saved. 
                           Default is '../data/processed/face_dataset'.
        move_images (bool): If True, the associated image files will be copied to the target directory. 
                            Default is True.
    Behavior:
        - For each dataset split ('train', 'val', 'test'), the function:
            1. Reads the corresponding XML file.
            2. Extracts bounding box and keypoint information for each image.
            3. Converts the extracted data into YOLO-compatible format.
            4. Saves the converted data as TXT files in the 'labels' subdirectory of the target path.
            5. Optionally moves the associated image files to the 'images' subdirectory of the target path.
    Notes:
        - Bounding box coordinates are normalized to the range [0, 1] relative to the image dimensions.
        - Keypoints are included only for inner edges of the eyebrows.
        - The function ensures that the target directory structure is created if it does not already exist.
    Prints:
        A success message for each dataset split indicating that the TXT files have been created and, 
        if applicable, that the images have been moved.
    Raises:
        - FileNotFoundError: If the XML file or image file does not exist in the specified source path.
        - ET.ParseError: If the XML file is malformed and cannot be parsed.
    """
    
    make_face_dataset_dir(target_path)
    
    for set in ['train', 'val', 'test']:
        xml_path = os.path.join(source_path, set, '{}.xml'.format(filename))
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for image in root.findall(".//image"):
            image_file = image.get("file")
            image_folder, image_file = image_file.rsplit('/', 1)
            source_image_file_path = os.path.join(source_path, set, image_folder, image_file)
            target_image_file_path = os.path.join(target_path, set, 'images', image_file)
            image_object = cv2.imread(source_image_file_path)
            H, W = image_object.shape[:2]

            if move_images:
                copyfile(source_image_file_path, target_image_file_path)

            txt_file_path = os.path.join(target_path, set, 'labels', os.path.splitext(image_file)[0] + '.txt')
            with open(txt_file_path, 'w') as txt_file:
                box = image.find("box")
                if box is not None:
                    xs = int(box.get("left"))
                    ys = int(box.get("top"))
                    xe = xs + int(box.get("width"))
                    ye = ys + int(box.get("height"))
                    
                    w = xe - xs
                    h = ye - ys
                                
                    xc = xs + w / 2
                    yc = ys + h / 2
                    
                    label = '0 {:.3f} {:.3f} {:.3f} {:.3f}'.format(xc / W, yc / H, w / W, h / H)
                    
                    for part in sorted(box.findall("part"), key=lambda p: int(p.get("name"))):
                        if int(part.get("name")) not in [21, 22]:
                            continue
                        
                        px = float(part.get("x")) / W
                        py = float(part.get("y")) / H

                        label += ' {:.3f} {:.3f}'.format(px, py)
                    
                    label += '\n'

                    txt_file.write(label)

        print(f"TXT files for labels in the '{set}' set are successfully created.{' Images are successfully moved.' if move_images else ''}")
