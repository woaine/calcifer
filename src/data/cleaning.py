import pandas as pd
import os
import xml.etree.ElementTree as ET
import cv2

from shutil import copyfile
from itertools import product

def clean_data(df:pd.DataFrame, data_group:str=None):
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
    for (set, data) in product(['train', 'val', 'test'], ['images', 'labels']):
        os.makedirs(os.path.join(target_path, set, data), exist_ok = True)

def XMLtoTXT(source_path: str = '../data/external/face_dataset', filename: str = 'facial_data', target_path: str = '../data/processed/face_dataset', move_images: bool = True):
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
