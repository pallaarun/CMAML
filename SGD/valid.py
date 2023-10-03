import pathlib
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

import os
import shutil

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

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)


def create_data_loaders(args):

    data = SliceData_validation(args.data_path)

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

    model = five_layerCNN_MAML(train_args).to(train_args.device)#.double() # double to make the weights in double since input type is double 
    if train_args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model


def run_unet(args, model, data_loader):

    reconstructions = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            input_mini_batch,target,slice_no_mini_batch,frame_no_mini_batch,one_task_mini_batch,_,fnames = data

            input_mini_batch = input_mini_batch.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)

            input_mini_batch = input_mini_batch.float()

            recons = model(input_mini_batch)

            recons = recons.squeeze(1).cpu().detach().numpy()

            for i in range(recons.shape[0]):

                reconstructions[fnames[i]].append((slice_no_mini_batch[i],frame_no_mini_batch[i],recons[i,:,:]))

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
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,help='Path to the U-Net model')

    parser.add_argument('--out-path', type=pathlib.Path, required=True,help='Path to save the reconstructions to')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')

    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')

    parser.add_argument('--data-dir',type=str,help='path to validation dataset')

    parser.add_argument('--task-strings',type=str,help='all tasks on which the validation is to be performed')
    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    
    all_tasks = args.task_strings.split(",")
    for one_task in all_tasks:

        args.degradation_type,args.degradation_amount = one_task.split("/")

        args.out_dir = args.out_path / args.degradation_type / "amount_{}".format(args.degradation_amount)

        args.data_path = args.data_dir+"/"+args.degradation_type+"/"+args.degradation_amount

        main(args)
