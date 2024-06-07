import cv2
import numpy as np
from typing import List

import sys
from scan import Scan, CHANNELS

DEFAULT_MAX_SCNAS_LOADED = 1000


class Dataset:
    __slots__ = ("scans", "path")
    def __init__(self) -> None:
        self.scans : List[Scan] = []
        self.path = None

    def load(self, path, number_of_scans : int = DEFAULT_MAX_SCNAS_LOADED) -> bool:
        new_scans : List[Scan]  = []
        for i in range(number_of_scans):
            try:
                scan = Scan(path, i)
            except ImportError as importError:
                break
            new_scans.append(scan)

        if len(new_scans) == 0:
            return False
        
        self.scans : List[Scan]  = new_scans
        self.path = path
        return True

    def save(self, save_folder : str = None) -> bool:
        if len(self.scans) == 0:
            False 
        
        if save_folder == None:
            save_folder = self.path
        
        for scan in self.scans:
            cv2.imwrite(f"{save_folder}/scan_{scan.scan_number:04d}_segmentation.png", scan.get_segmentation())
            cv2.imwrite(f"{save_folder}/scan_{scan.scan_number:04d}_colored_segmentation.png", scan.get_masked_channel(CHANNELS.TEXTURE))

        print(f"saved dataset to: {save_folder}")

        return True
    
    def get_number_of_scans(self):
        return len(self.scans)

    def get_scan(self, scan_index : int) -> Scan:
        if scan_index >= len(self.scans):
            return
        return self.scans[scan_index]

if __name__ == "__main__":
    path = "C:/MATFYZ/bakalarka/datasets/SAM_data/2case"
    dataset = Dataset()
    dataset.load(path)

    for i in range(dataset.get_number_of_scans()):
        scan = dataset.get_scan(i)
        cv2.imshow(scan.scan_name + str(scan.scan_number), scan.get_channel(CHANNELS.TEXTURE))
        cv2.waitKey(1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()





