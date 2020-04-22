"""
dir_loader.py    Load files from directory, with filename
"""

import os
import cv2
from pathlib import Path
from torchvision import transforms
from PIL import Image

SUFFICES = ['.jpg', '.jpeg', '.png', '.bmp']

class Dirloader:
    def __init__(
            self, top, reftop,
            suffices=SUFFICES):
        self._top = top
        self._reftop = reftop
        self._suffices = suffices

    def __iter__(self):
        refp = Path(self._reftop)
        for dirpath, dirnames, filenames in os.walk(self._top):
            for filename in filenames:
                if Path(filename).suffix.lower() in self._suffices:
                    fullpath = Path(dirpath) / Path(filename)
                    yield fullpath, refp / fullpath.name

class ListLoader:
    def __init__(
            self, top, reftop,
            lst,
            suffices=SUFFICES):
        self._top = top
        self._reftop = reftop
        self._lst = lst
        self._suffices = suffices

    def __iter__(self):
        topp = Path(self._top)
        refp = Path(self._reftop)
        for row in self._lst:
            ref_fullpath = refp / row[0]
            fullpath = topp / row[1]
            yield fullpath, ref_fullpath

if __name__ == '__main__':
    loader = Dirloader('.')
    for x, im in loader:
        print(x, im.shape)


