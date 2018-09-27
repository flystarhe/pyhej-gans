import json
import codecs
import random
import pydicom
import numpy as np
import torch


def dicom_read_ct(file_path):
    plan = pydicom.dcmread(file_path, force=True)
    dtype = "" if getattr(plan, "PixelRepresentation", 0) else "u"
    dtype = "{}int{}".format(dtype, getattr(plan, "BitsAllocated", 8))
    data = np.frombuffer(plan.PixelData, dtype=dtype)
    data = data.reshape((-1, plan.Rows, plan.Columns))[:1]
    rescale_slope = getattr(plan, "RescaleSlope", 1)
    rescale_intercept = getattr(plan, "RescaleIntercept", 0)
    r_ct = data * rescale_slope + rescale_intercept
    return r_ct.clip(-1000, None)


def dicom_read_ac(file_path):
    # transfer the hunsfield units to attenuation coefficents
    r_ct = dicom_read_ct(file_path)
    # 0.17 is the attenuation coef (1/cm) of water at 100 keV
    r_ac = 0.17 * r_ct / 1000 + 0.17
    return r_ac.clip(0, None)


def ac2ct(ac, window=None):
    # window is tuple, such as `(center, width)`
    ct = (ac - 0.17) * 1000 / 0.17
    if window is not None:
        center, width = window
        ct = (ct - center + width / 2) / width * 255
        ct = ct.clip(0, 255).astype("uint8")
    return ct


def load_dicom(file_path):
    arr = dicom_read_ct(file_path) + 1000
    arr = (arr / arr.max() * 2 - 1).astype(np.float32)
    return torch.from_numpy(arr)


class Dataset(object):
    def __init__(self, json_file, aligned=True):
        # dataroot such as **/*.json
        self.aligned = aligned
        with codecs.open(json_file, "r", "utf-8") as reader:
            self.data = json.load(reader)
        self.A_size = len(self.data["A"])
        self.B_size = len(self.data["B"])

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):
        A_path = self.data["A"][index % self.A_size]
        index = index % self.B_size if self.aligned else random.randint(0, self.B_size - 1)
        B_path = self.data["B"][index]
        return load_dicom(A_path), load_dicom(B_path)
