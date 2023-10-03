import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature
import os

from utils import CreateZeroFilledImageFn

class SliceData_not_onthefly(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self,root,artifact_type,train_valid_support_or_query): # acc_factor can be passed here and saved as self variable
        self.examples = []
        self.root = root
        newroot = os.path.join(root,artifact_type,train_valid_support_or_query)
        #print(newroot)
        files = list(pathlib.Path(newroot).iterdir())
        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                gt_vol = hf['gt']
                if len(gt_vol.shape) == 3: ## Not used anymore because the motion artifact is now 4d
                    frame_no = 0
                    #acc_factor = float(acc_factor[:-1].replace("_","."))
                    self.examples = [(fname,slice_no,frame_no,artifact_type) for slice_no in range(num_slices)]
                
                elif len(gt_vol.shape) == 4: ## Everything goes through this logic, both spatial and motion
                    num_slices = gt_vol.shape[2]
                    num_frames = gt_vol.shape[3]

                    middle_slice_no = num_slices//2
                    blank_frames = []
                    for frame_idx in range(0,num_frames):
                        if np.abs(np.max(hf["crp_inp"][:,:,middle_slice_no,frame_idx])) == 0:
                            blank_frames.append(frame_idx)

                    for frame_no in range(num_frames):
                        if frame_no not in blank_frames:
                            for slice_no in range(num_slices):
                                self.examples.append((fname,slice_no,frame_no,artifact_type))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname,slice_no,frame_no,artifact_type = self.examples[i]

        with h5py.File(fname, 'r') as data:

            no_of_dims = len(data["gt"].shape)
            
            if no_of_dims == 3: ## Not used anymore because the motion artifact is now 4d

                input_img  = np.abs(data["crp_inp"][:,:,slice_no]) #Artifact slices are complex datatype
                target = data['crp_gt'][:,:,slice_no].astype(np.float64)# converting to double
            
            elif no_of_dims == 4: ## Everything goes through this logic, both spatial and motion

                input_img  = np.abs(data["crp_inp"][:,:,slice_no,frame_no])
                target = data['crp_gt'][:,:,slice_no,frame_no].astype(np.float64)# converting to double

            return torch.from_numpy(input_img),torch.from_numpy(target),artifact_type,str(fname)

class SliceData_onthefly(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self,root,artifact_type,train_valid_support_or_query): # acc_factor can be passed here and saved as self variable
        self.examples = []
        self.root = root
        newroot = os.path.join(root,artifact_type,train_valid_support_or_query)
        #print(newroot)
        files = list(pathlib.Path(newroot).iterdir())
        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                gt_vol = hf['gt']
                if len(gt_vol.shape) == 3: ## Not used anymore because the motion artifact is now 4d
                    frame_no = 0
                    #acc_factor = float(acc_factor[:-1].replace("_","."))
                    self.examples = [(fname,slice_no,frame_no,artifact_type) for slice_no in range(num_slices)]

                elif len(gt_vol.shape) == 4: ## Everything goes through this logic, both spatial and motion
                    num_slices = gt_vol.shape[2]
                    num_frames = gt_vol.shape[3]

                    middle_slice_no = num_slices//2
                    blank_frames = []
                    for frame_idx in range(0,num_frames):
                        if np.abs(np.max(hf["crp_inp"][:,:,middle_slice_no,frame_idx])) == 0:
                            blank_frames.append(frame_idx)

                    for frame_no in range(num_frames):
                        if frame_no not in blank_frames:
                            for slice_no in range(num_slices):
                                self.examples.append((fname,slice_no,frame_no,artifact_type))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__

        fname,slice_no,frame_no,artifact_type = self.examples[i]

        with h5py.File(fname, 'r') as data:

            artifact_name,artifact_amount = artifact_type.split("/")[0],int(artifact_type.split("/")[1])

            target = data['crp_gt'][:,:,slice_no,frame_no].astype(np.float64)# converting to double
            if artifact_name != "Undersampling":
                input_img  = np.abs(data["crp_inp"][:,:,slice_no,frame_no])
            elif artifact_name == "Undersampling":
                input_img = CreateZeroFilledImageFn(target,artifact_amount,mask_type = "cartesian")

            return torch.from_numpy(input_img),torch.from_numpy(target),artifact_type,str(fname)


class taskdataset(Dataset):
    """
    A PyTorch Dataset that provides different tasks.
    """

    def __init__(self,task_list):

        self.task_list = task_list

    def __len__(self):
        
        return len(self.task_list)

    def __getitem__(self,index):
        
        task_mini_batch = self.task_list[index]

        return task_mini_batch


class slice_indices(Dataset):
    """
    A PyTorch Dataset that provides indices for a task based on the total amount of data.
    """

    def __init__(self,datapoints):

        self.datapoints = datapoints

    def __len__(self):

        return len(self.datapoints)

    def __getitem__(self,index):

        datapoints_mini_batch = self.datapoints[index]

        return datapoints_mini_batch


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



class SliceData_test(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self,root,artifact_type,train_valid_support_or_query): # acc_factor can be passed here and saved as self variable
        self.examples = []
        self.root = root
        newroot = os.path.join(root,artifact_type,train_valid_support_or_query)
        #print(newroot)
        files = list(pathlib.Path(newroot).iterdir())
        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                gt_vol = hf['gt']
                if len(gt_vol.shape) == 3: ## Not used anymore because the motion artifact is now 4d
                    frame_no = 0
                    #acc_factor = float(acc_factor[:-1].replace("_","."))
                    self.examples = [(fname,slice_no,frame_no,artifact_type) for slice_no in range(num_slices)]
                
                elif len(gt_vol.shape) == 4: ## Everything goes through this logic, both spatial and motion
                    num_slices = gt_vol.shape[2]
                    num_frames = gt_vol.shape[3]

                    for frame_no in range(num_frames):
                        for slice_no in range(num_slices):
                            self.examples.append((fname,slice_no,frame_no,artifact_type))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname,slice_no,frame_no,artifact_type = self.examples[i]

        with h5py.File(fname, 'r') as data:

            no_of_dims = len(data["gt"].shape)
            
            if no_of_dims == 3: ## Not used anymore because the motion artifact is now 4d

                input_img  = np.abs(data["crp_inp"][:,:,slice_no]) #Artifact slices are complex datatype
                target = data['crp_gt'][:,:,slice_no].astype(np.float64)# converting to double
            
            elif no_of_dims == 4: ## Everything goes through this logic, both spatial and motion

                input_img  = np.abs(data["crp_inp"][:,:,slice_no,frame_no])
                target = data['crp_gt'][:,:,slice_no,frame_no].astype(np.float64)# converting to double

            return torch.from_numpy(input_img),torch.from_numpy(target),artifact_type,slice_no,frame_no,str(fname)
