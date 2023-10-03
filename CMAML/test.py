import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SliceData_test,taskdataset
from models import five_layerCNN_MAML
import h5py
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import time
from torch.nn import functional as F
import logging
import shutil
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_reconstructions(reconstructions,out_dir,h5_file_name):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (Model prediction itself of size batchxheight,weight)
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    #out_dir = pathlib.Path(out_dir_str)
    #out_dir.mkdir(exist_ok=True)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    file_name = h5_file_name.split("/")[-1]
    with h5py.File(out_dir+file_name, 'w') as f:
        f.create_dataset('reconstruction',data = reconstructions)

        
        
def CreateLoadersForTestTasks(args):
    
    support_loaderdict = {}
    query_loaderdict = {}

    full_tasks = args.test_task_strings.split(",")

    test_path = args.test_path
    for one_task in full_tasks:

        support_task_dataset = SliceData_test(test_path,one_task,"valid_support")
        query_task_dataset = SliceData_test(test_path,one_task,"valid_query")

        support_task_loader = DataLoader(dataset = support_task_dataset, batch_size = args.test_support_batch_size,shuffle = True)
        query_task_loader = DataLoader(dataset = query_task_dataset, batch_size = len(query_task_dataset),shuffle = False)

        loaderkey = one_task

        support_loaderdict[loaderkey] = support_task_loader

        query_loaderdict[loaderkey] = query_task_loader
    return support_loaderdict,query_loaderdict


def load_model(checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    train_args = checkpoint['args']

    model = five_layerCNN_MAML(train_args).to(train_args.device).double() # to make the weights in double since input type is double 
    model.load_state_dict(checkpoint['model'])

    return model


def test_time_inference(args,model,test_task_loader,test_support_loader,test_query_loader,writer):
    
    start = start_iter = time.perf_counter()
    
    for iter,test_task_mini_batch in enumerate(test_task_loader):
        
        test_meta_loss = 0
        
        for test_task in test_task_mini_batch:
            
            base_weights = list(model.parameters())
            clone_weights = [w.clone() for w in base_weights]
            
            adapt_flag = False
            no_of_gd_steps = 0
            while True:

                for gd_steps,test_spt_data_batch in enumerate(test_support_loader[test_task]):
                    no_of_gd_steps = no_of_gd_steps+1 # gd_steps will reset to 0 once the loader exhausts. Hence we introduce another counter no_of_gd_steps to get the actual number of times the loader is enumerated

                    model.train()

                    test_spt_inp_imgs = test_spt_data_batch[0].unsqueeze(1).to(args.device)
                    test_spt_tgt_imgs = test_spt_data_batch[1].unsqueeze(1).to(args.device)

                    test_spt_pred = model.adaptation(test_spt_inp_imgs,clone_weights)
                    test_spt_loss = F.l1_loss(test_spt_pred,test_spt_tgt_imgs)
                    
                    if no_of_gd_steps == 1:
                        best_adapt_loss = test_spt_loss.item()
                        best_adapt_model_weights = [w.clone() for w in clone_weights]
                    elif best_adapt_loss > test_spt_loss.item():
                        best_adapt_loss = test_spt_loss.item()
                        best_adapt_model_weights = [w.clone() for w in clone_weights]
                    
                    clone_grads = torch.autograd.grad(test_spt_loss, clone_weights)
                    clone_weights = [w-args.test_adapt_lr*g for w, g in zip(clone_weights,clone_grads)]
                    
                    if iter % args.report_interval == 0:
                        logging.info(
                                f'Adaptation_step = [{no_of_gd_steps:3d}/{args.no_of_test_adaptation_steps:3d}] '
                                f'Adaptation loss = {test_spt_loss.item():.4g} '
                                f'Time = {time.perf_counter() - start_iter:.4f}s',
                                )
                    start_iter = time.perf_counter()

                    writer.add_scalar(test_task+'_per_adaptation_step_spt_loss',test_spt_loss.item(),no_of_gd_steps)

                    if no_of_gd_steps >= args.no_of_test_adaptation_steps:
                        adapt_flag = True
                        break
                    
                if adapt_flag:
                    print("Adaptation stopped after gd_steps {}".format(no_of_gd_steps))
                    break

            for _,test_qry_data_batch in enumerate(test_query_loader[test_task]):

                model.eval()
                test_qry_inp_imgs = test_qry_data_batch[0]
                test_qry_tgt_imgs = test_qry_data_batch[1]
                test_qry_slice_numbers = test_qry_data_batch[3]
                test_qry_frame_numbers = test_qry_data_batch[4]
                test_qry_file_names = test_qry_data_batch[5]
                break

            unique_file_names = np.unique(test_qry_file_names)
            dict_file_names = {}
            for one_file_name in unique_file_names:
                dict_file_names[one_file_name] = []
                for idx,file_name in enumerate(test_qry_file_names):
                    if file_name == one_file_name:
                        dict_file_names[one_file_name].append([test_qry_slice_numbers[idx],test_qry_frame_numbers[idx],test_qry_inp_imgs[idx,:,:],test_qry_tgt_imgs[idx,:,:]])

            for one_file_name,slice_preds in dict_file_names.items():
                max_slices = max(slice_preds,key=lambda x: x[0])[0]+1
                max_frames = max(slice_preds,key=lambda x: x[1])[1]+1
                height,width = slice_preds[0][2].shape
                np_inp = np.zeros([height,width,max_slices,max_frames])
                np_tgt = np.zeros([height,width,max_slices,max_frames])
                np_pred = np.zeros([height,width,max_slices,max_frames])

                for slice_no,frame_no,inp_array,tgt_array in slice_preds:
                    np_inp[:,:,slice_no,frame_no] = inp_array
                    np_tgt[:,:,slice_no,frame_no] = tgt_array

                    with torch.no_grad(): # this is added to ensure that gradients do not occupy the gpu mem
                        test_qry_pred = model.adaptation(inp_array.unsqueeze(0).unsqueeze(1).to(args.device),best_adapt_model_weights)
                        test_qry_loss = F.l1_loss(test_qry_pred,tgt_array.unsqueeze(0).unsqueeze(1).to(args.device))

                        test_meta_loss = test_meta_loss+test_qry_loss
                    np_pred[:,:,slice_no,frame_no] = test_qry_pred[0,0,:,:].cpu().detach().numpy()

                current_task_path = test_task.split("/")
                task_specific_path = str(args.results_dir) + "/" + current_task_path[0] + "/"+"amount_"+current_task_path[1] + "/"

                save_reconstructions(np_pred,task_specific_path,one_file_name)

            test_meta_loss = test_meta_loss/test_qry_inp_imgs.shape[0]
            writer.add_scalar(test_task+'_qry_loss',test_meta_loss.item(),1)
        
        logging.info('Query loss for task {}:{}'.format(test_task,test_meta_loss.item()))
    
    return np_pred,time.perf_counter()-start


def main(args):
    
    test_support_loader,test_query_loader = CreateLoadersForTestTasks(args)

    test_task_strings = args.test_task_strings.split(",")

    test_task_dataset_instantiation = taskdataset(test_task_strings)

    test_task_batch_size = args.test_task_batch_size
    test_task_loader = DataLoader(dataset = test_task_dataset_instantiation, batch_size = test_task_batch_size,shuffle = False)

    print ("Task, Support and Query loaders are initialized")
    
    model = load_model(args.checkpoint)
    
    writer = SummaryWriter(log_dir=str(args.tensorboard_summary_dir))
    
    reconstructions,_ = test_time_inference(args,model,test_task_loader,test_support_loader,test_query_loader,writer)

    
    
def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')

    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the trained model')
    parser.add_argument('--results_dir', type=pathlib.Path, required=True,
                        help='Base path to save the reconstructions to')
    parser.add_argument('--tensorboard_summary_dir', type=pathlib.Path, required=True,
                        help='Path to write the summary files')
    
    parser.add_argument('--test_task_batch_size',default=1,type=int, help='Test task batch size')
    parser.add_argument('--test_support_batch_size', default=47, type=int, help='Support batch size')
    
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--test_path',type=pathlib.Path ,required=True, help = 'path to test dataset')

    parser.add_argument('--no_of_test_adaptation_steps', type=int, default=3,
                        help='Number of adaptation steps during meta-training stage')
    
    parser.add_argument('--test_task_strings',type=str,help='All the test tasks')
    
    parser.add_argument('--test_adapt_lr', type=float, default=0.001, help='Task-specific Learning rate')

    parser.add_argument('--report-interval', type=int, default=1, help='Period of loss reporting')
    return parser

if __name__ == '__main__':
    #args = create_arg_parser().parse_args(sys.argv[1:])
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
