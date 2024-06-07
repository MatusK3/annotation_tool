# import torch
# import torchvision
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" 
import cv2
# import sys
# sys.path.append("..")
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from skimage import exposure
from enum import Enum
from typing import List


class CHANNELS(Enum):
    NORMALS, DEPTH_MAP, TEXTURE, COORDINATIONS, RAW_COORDINATIONS = range(5)

class CHANNELS_SUFFIC_TYPES(Enum):
    NORMALS, DEPTH_MAP, NORMAL_MAP_X, NORMAL_MAP_Y, NORMAL_MAP_Z, POINT_CLOUD_X, POINT_CLOUD_Y, POINT_CLOUD_Z, TEXTURE_8_BIT, TEXTURE_24_BIT = range(10)

CHANNELS_FILES_SUFFIXES = {
    CHANNELS_SUFFIC_TYPES.NORMALS : ".png",
    CHANNELS_SUFFIC_TYPES.DEPTH_MAP : "_IMG_DepthMap.tif",
    CHANNELS_SUFFIC_TYPES.NORMAL_MAP_X : "_IMG_NormalMap_X.tif",
    CHANNELS_SUFFIC_TYPES.NORMAL_MAP_Y : "_IMG_NormalMap_Y.tif",
    CHANNELS_SUFFIC_TYPES.NORMAL_MAP_Z : "_IMG_NormalMap_Z.tif",
    CHANNELS_SUFFIC_TYPES.POINT_CLOUD_X : "_IMG_PointCloud_X.tif",
    CHANNELS_SUFFIC_TYPES.POINT_CLOUD_Y : "_IMG_PointCloud_Y.tif",
    CHANNELS_SUFFIC_TYPES.POINT_CLOUD_Z : "_IMG_PointCloud_Z.tif",
    CHANNELS_SUFFIC_TYPES.TEXTURE_8_BIT : "_IMG_Texture_8Bit.png",
    CHANNELS_SUFFIC_TYPES.TEXTURE_24_BIT : "_IMG_Texture_R.tif"
}



class Scan:
    __slots__ = ("channels", "scan_name", "scan_number", "file_path", "half_resolution", "masks", "masks_colors")
    
    def __init__(self, dataset_path, scan_number, half_resolution=False) -> None:
        self.channels = dict()
        self.scan_name = dataset_path.split("/")[-1]
        self.scan_number = scan_number
        self.file_path = f"{dataset_path}/scan_{scan_number:04d}"
        self.half_resolution = half_resolution

        self.masks = []
        self.masks_colors = [np.random.randint(0, 256, size=(1, 3), dtype=np.uint8) for i in range(10000)]
        

        self.load()

    def _check_files_integrity(self) -> None:
        missing_files = []

        for suffix_type in [
            CHANNELS_SUFFIC_TYPES.NORMAL_MAP_X, CHANNELS_SUFFIC_TYPES.NORMAL_MAP_Y, CHANNELS_SUFFIC_TYPES.NORMAL_MAP_Z,
            CHANNELS_SUFFIC_TYPES.DEPTH_MAP, CHANNELS_SUFFIC_TYPES.TEXTURE_8_BIT
        ]:
            path = f"{self.file_path}{CHANNELS_FILES_SUFFIXES[suffix_type]}"
            if not os.path.exists(path):
                missing_files.append(path)
        
        if len(missing_files) != 0:
            raise ImportError('file integrity vaiolated: ', *missing_files)

    def _load_normals_xyz(self) -> np.ndarray:
        normals_x = cv2.imread(f"{self.file_path}{CHANNELS_FILES_SUFFIXES[CHANNELS_SUFFIC_TYPES.NORMAL_MAP_X]}", cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        normals_y = cv2.imread(f"{self.file_path}{CHANNELS_FILES_SUFFIXES[CHANNELS_SUFFIC_TYPES.NORMAL_MAP_Y]}", cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        normals_z = cv2.imread(f"{self.file_path}{CHANNELS_FILES_SUFFIXES[CHANNELS_SUFFIC_TYPES.NORMAL_MAP_Z]}", cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)

        normals = cv2.merge((normals_x, normals_y, normals_z))
        linalg = np.linalg.norm(normals, axis=2, keepdims=True)
        linalg[linalg == 0] = 0.000000001
        normals /= linalg

        normals = cv2.normalize(normals, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        if(self.half_resolution):
            return cv2.resize(normals, (0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_NEAREST)
        return normals

    def _load_depth(self) -> np.ndarray:
        depth_map = cv2.imread(f"{self.file_path}{CHANNELS_FILES_SUFFIXES[CHANNELS_SUFFIC_TYPES.DEPTH_MAP]}", cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        
        depth_map_equalized = exposure.equalize_hist(depth_map)
        normalized_equalized_depth_map = cv2.normalize(depth_map_equalized, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        inverted = cv2.bitwise_not(normalized_equalized_depth_map)

        normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        _,otsu = cv2.threshold(normalized_depth_map,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        masked = cv2.bitwise_and(inverted, inverted, mask=otsu)

        merged = cv2.merge((masked, masked, masked))
        if(self.half_resolution):
            return cv2.resize(merged, (0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_NEAREST)
        return merged

    def _load_texture(self) -> np.ndarray:
        texture = cv2.imread(f"{self.file_path}{CHANNELS_FILES_SUFFIXES[CHANNELS_SUFFIC_TYPES.TEXTURE_8_BIT]}", cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if(self.half_resolution):
            return cv2.resize(texture, (0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_NEAREST)
        return texture

    def _load_segmantation(self):
        default_segmentation_file_path = f"{self.file_path}_segmentation.png"

        if not os.path.exists(default_segmentation_file_path):
            return
        
        masks_image = cv2.imread(default_segmentation_file_path, cv2.IMREAD_GRAYSCALE)
        if self.half_resolution:
            masks_image = cv2.resize(masks_image, (0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_NEAREST)

        for i in range(1, 255):
            mask = masks_image == i
            if np.any(mask):
                self.add_mask(mask)
            else:
                break

    def load(self) -> None:
        self._check_files_integrity()

        self.channels[CHANNELS.NORMALS] = self._load_normals_xyz()
        self.channels[CHANNELS.DEPTH_MAP] = self._load_depth()
        self.channels[CHANNELS.TEXTURE] = self._load_texture()

        self._load_segmantation()


    def set_mask_colors(self, colors):
        self.masks_colors = colors

    def get_mask_by_coords(self, pos):
        x, y = pos
        for mask in self.masks:
            if mask[x, y]:
                return mask

    def add_mask(self, mask, label=""):
        self.masks.append([mask, label])
        self.masks.sort(key=lambda m: np.sum(m[0]), reverse=True)

    def add_masks(self, masks, labels):
        self.masks = list(zip(masks, labels))    

    
    def change_mask_label(self, index, label):
        self.masks[index][1] = label

    # def set_mask(self, index, mask):
    #     self.masks[index] = mask
    #     self.masks.sort(key=lambda m: np.sum(m), reverse=True)
  
    def dellete_mask(self, index):
        if index < 0 or index >= len(self.masks):
            return

        self.masks.pop(index)

    def delete_masks(self):
        self.masks = []

    def get_masked_channel(self, scan_channel : CHANNELS = CHANNELS.TEXTURE, selected_masks = set()):
        segmented_image = self.get_channel(scan_channel)
    
        for index in range(len(self.masks)):
            segmented_image[self.masks[index][0]] = self.masks_colors[index]

        return segmented_image
    
    def get_segmentation(self):
        segmented_image = np.zeros(self.channels[CHANNELS.TEXTURE].shape[:2])

        for index in range(len(self.masks)):
            segmented_image[self.masks[index][0]] = index+1

        return segmented_image

    def get_channel(self, channel) -> np.ndarray:
        channel = self.channels.get(channel, None)
        if channel is None:
            return
        return channel.copy()
    
    def get_shape(self):
        return self.channels.get(CHANNELS.TEXTURE).shape
    


    

DATASET_PATH = "C:/MATFYZ/bakalarka/datasets/SAM_data"
if __name__ == "__main__":

    # scan_files = ["2case" "metrie", "tirepicker"]
    scan_files = ["tirepicker"]#, "metrie", "tirepicker"]
    scans : List[Scan] = []
    # for file in scan_files:
    #     for i in range(10):
    #         path = f"{DATASET_PATH}/{file}"
    #         scans.append(Scan(path, i))
    #         scans[-1].show_channels()
    for file in scan_files:
        path = f"{DATASET_PATH}/{file}"
        for i in range(6):
            scans.append(Scan(path, i))
    print("scans loaded")

    cv2.waitKey(0)
    cv2.destroyAllWindows()