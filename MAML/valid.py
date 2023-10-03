import os
import pathlib
import shutil
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SliceData_validation
from models import five_layerCNN_MAML 
import h5py
from tqdm import tqdm

def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
#    out_dir.mkdir(exist_ok=True)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for fname, recons in reconstructions.items():
        patient_name = fname.split("/")[-1]
        final_path = pathlib.Path.joinpath(out_dir,patient_name)
        with h5py.File(final_path, 'w') as f:
            f.create_dataset('reconstruction', data=recons)


def create_data_loaders(args):

    data = SliceData_validation(args.data_path,args.degradation_amount,args.degradation_type)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )

    return data_loader


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    train_args = checkpoint['args']
    model = five_layerCNN_MAML(train_args).to(train_args.device)#.double() # to make the weights in double since input type is double 
    #print(model)
    if train_args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model


def run_unet(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            input_img,target,artifact_type,slices,frames,fnames = data

            input_img = input_img.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)

            input_img = input_img.float()

            recons = model(input_img)  

            recons = recons.to('cpu').squeeze(1)

            for i in range(recons.shape[0]):
                recons[i] = recons[i]
                reconstructions[fnames[i]].append((slices[i].numpy(),frames[i].numpy(),recons[i].numpy()))
    
    new_recons = {}
    for fname,slice_preds in reconstructions.items():
        max_slices = max(slice_preds,key=lambda x: x[0])[0]+1
        max_frames = max(slice_preds,key=lambda x: x[1])[1]+1
        height,width = slice_preds[0][2].shape
        np_preds = np.zeros([height,width,max_slices,max_frames])
        
        for slice_no,frame_no,prediction_array in slice_preds:
            np_preds[:,:,slice_no,frame_no] = prediction_array
        new_recons[fname] = np_preds

    return new_recons


def main(args):
    
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)
    reconstructions = run_unet(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-path', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--data-dir',type=str,help='path to validation dataset')

    parser.add_argument('--valid_task_strings',type=str,help='validation task strings')
    parser.add_argument('--flag',type=str,help='query or support data')

    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    valid_task_strings = args.valid_task_strings.split(",")
    for task in valid_task_strings:
        args.degradation_type,args.degradation_amount = task.split("/")

        args.out_dir = args.out_path / args.degradation_type / 'amount_{}'.format(args.degradation_amount)
        args.data_path = args.data_dir+'/'+args.degradation_type+'/'+args.degradation_amount+'/'+args.flag
        main(args)
