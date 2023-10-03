import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature
import os

class SliceData_validation(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root,degradation_amount,degradation_type):
    #def __init__(self, root,acc_factor,dataset_type):

        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                crp_inp = hf['crp_inp']
                num_slices,num_frames = crp_inp.shape[2],crp_inp.shape[3]
                for slice_no in range(num_slices):
                    for frame_no in range(num_frames):
                        self.examples.append((fname,slice_no,frame_no,degradation_amount,degradation_type))

          

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname,slice_no,frame_no,degradation_amount,degradation_type = self.examples[i]
    
        with h5py.File(fname, 'r') as data:

            input_img  = np.abs(data["crp_inp"][:,:,slice_no,frame_no])
            target = data['crp_gt'][:,:,slice_no,frame_no].astype(np.float64)# converting to double

            return torch.from_numpy(input_img),torch.from_numpy(target),degradation_type,slice_no,frame_no,str(fname)
