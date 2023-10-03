import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import numpy as np
from runstats import Statistics
#from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters import laplace
from tqdm import tqdm

import os
import pandas as pd
from math import nan, isnan
import matplotlib.pyplot as plt

# adding hfn metric 
def hfn(gt,pred):

    hfn_total = []

    for frame_no in range(gt.shape[3]):
        for slice_no in range(gt.shape[2]):
            gt_slice = gt[:,:,slice_no,frame_no]
            pred_slice = pred[:,:,slice_no,frame_no]

            pred_slice[pred_slice<0] = 0 #bring the range to 0 and 1.
            pred_slice[pred_slice>1] = 1

            gt_slice_laplace = laplace(gt_slice)        
            pred_slice_laplace = laplace(pred_slice)

            hfn_slice = np.sum((gt_slice_laplace - pred_slice_laplace) ** 2) / np.sum(gt_slice_laplace **2)
            hfn_total.append(hfn_slice)

    return np.mean(hfn_total)


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    #return compare_ssim(
    #    gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    #)

    _,_,no_of_slices,no_of_frames = gt.shape
    min_val = int(min(no_of_frames,no_of_slices))
    if min_val > 2:
        if min_val%2 == 0:
            win_size = min_val-1
        elif min_val%2 == 1:
            win_size = min_val

        return structural_similarity(gt,pred,multichannel=True, data_range=gt.max(),win_size = win_size)
    elif min_val <= 2:
        return 0

def ssim_slicewise(gt,pred):
    """ Compute ssim for a 2D slice that is used in measures_csv.py. """
    return structural_similarity(gt,pred)


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
    HFN=hfn
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }


    '''
    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )
    '''

    def get_report(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )


def get_metric_report(per_vol_metric_dict):
    metric_str = ""
    for metric_name,metric_list in per_vol_metric_dict.items():

        mean_metric = np.average(metric_list)
        std_metric = np.std(metric_list)

        metric_str = metric_str+"{} = {:.4g} +/- {:.4g} ".format(metric_name,mean_metric,std_metric)

    return metric_str

def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

    for tgt_file in args.target_path.iterdir():
        #print (tgt_file)
        with h5py.File(tgt_file) as target, h5py.File(args.predictions_path / tgt_file.name) as recons:

            target = target[recons_key]
            target = np.array(target)

            recons = np.array(recons['reconstruction'])
            
            # recons = np.transpose(recons,[1,2,0])

            metrics.push(target, recons)
            
    return metrics

def evaluate_slicewise(args,recons_key,metrics_names):

    metric_dict = {metric_name:[] for metric_name in metrics_names}
    if "fname" in metrics_names:
        del metric_dict["fname"]

    for tgt_file in args.target_path.iterdir():

        one_vol_psnr = 0
        one_vol_ssim = 0
        one_vol_mse = 0
        one_vol_nmse = 0

        with h5py.File(tgt_file) as target, h5py.File(args.predictions_path / tgt_file.name) as recons:

            target = target[recons_key]
            target = np.array(target)

            recons = np.array(recons['reconstruction'])

            _,_,no_of_slices,no_of_frames = target.shape

            count = 0
            for frame_no in range(no_of_frames):
                for slice_no in range(no_of_slices):
                    tgt_slice = target[:,:,slice_no,frame_no]
                    pred_slice = recons[:,:,slice_no,frame_no]

                    if np.max(tgt_slice) != 0:

                        count = count+1

                        one_vol_psnr = one_vol_psnr+psnr(tgt_slice,pred_slice)
                        one_vol_ssim = one_vol_ssim+structural_similarity(tgt_slice,pred_slice)
                        one_vol_mse = one_vol_mse+mse(tgt_slice,pred_slice)
                        one_vol_nmse = one_vol_nmse+nmse(tgt_slice,pred_slice)

        metric_dict[metrics_names[0]].append(one_vol_psnr/count)
        metric_dict[metrics_names[1]].append(one_vol_ssim/count)
        metric_dict[metrics_names[2]].append(one_vol_mse/count)
        metric_dict[metrics_names[3]].append(one_vol_nmse/count)

    return metric_dict


def evaluate_slicewise_for_boxplot(args,recons_key,metrics_names):
    metric_dict = {metric_name:[] for metric_name in metrics_names}

    for tgt_file in args.target_path.iterdir():
        with h5py.File(tgt_file) as target, h5py.File(args.predictions_path / tgt_file.name) as recons:

            target = target[recons_key]
            target = np.array(target)

            recons = np.array(recons['reconstruction'])

            _,_,no_of_slices,no_of_frames = target.shape

            for frame_no in range(no_of_frames):
                for slice_no in range(no_of_slices):
                    tgt_slice = target[:,:,slice_no,frame_no]
                    pred_slice = recons[:,:,slice_no,frame_no]

                    metric_dict[metrics_names[0]].append(psnr(tgt_slice,pred_slice))
                    metric_dict[metrics_names[1]].append(structural_similarity(tgt_slice,pred_slice))
                    metric_dict[metrics_names[2]].append(mse(tgt_slice,pred_slice))
                    metric_dict[metrics_names[3]].append(nmse(tgt_slice,pred_slice))
                    metric_dict[metrics_names[4]].append(tgt_file.name)

    return metric_dict

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data-dir', type=pathlib.Path, required=True,help='Path to the ground truth data')
    parser.add_argument('--base-path', type=pathlib.Path, required=True,help='Path to reconstructions')

    parser.add_argument('--report-path', type=pathlib.Path, required=True,help='Path to save metrics')

    parser.add_argument('--task_strings', type=str, default='',help='Put all the tasks')

    parser.add_argument('--flag', type=str, default='',help='Valid_query')
    
    parser.add_argument('--results-type', type=str, default='',help='Adapted/Unadapted Seen/Unseen results')

    args = parser.parse_args()
    
    if not os.path.exists(args.report_path):
        os.makedirs(args.report_path)

    metrics_names = ["PSNR","SSIM","mse","nmse","fname"]
    metrics_dict = {}

    all_tasks = args.task_strings.split(",")
    for one_task in tqdm(all_tasks):

        args.degradation_type,args.degradation_amount = one_task.split("/")

        args.target_path = pathlib.Path.joinpath(args.data_dir,one_task+'/'+args.flag)

        args.predictions_path = pathlib.Path.joinpath(args.base_path,args.results_type+'/'+args.degradation_type+'/amount_'+args.degradation_amount)

        recons_key = 'crp_gt'

        # metrics = evaluate(args,recons_key)

        # metrics_report = metrics.get_report()

        per_vol_metrics_dict = evaluate_slicewise(args,recons_key,metrics_names)
        metrics_report = get_metric_report(per_vol_metrics_dict)

        with open(args.report_path / 'report_{}_amount_{}.txt'.format(args.degradation_type,args.degradation_amount),'w') as f:
            f.write(metrics_report)

        one_task_metric_dict = evaluate_slicewise_for_boxplot(args,recons_key,metrics_names)
        metrics_dict[one_task] = one_task_metric_dict

    args.performance_visulization_path = args.base_path / "performance_visualization"
    if not os.path.exists(args.performance_visulization_path):
        os.makedirs(args.performance_visulization_path)

    for metric_name in metrics_names:
        if metric_name != "fname":
            bp_dict = {}

            for one_task,metric_value in metrics_dict.items():
                metric_list_with_NaN = metric_value[metric_name]
                metric_list = [x for x in metric_list_with_NaN if isnan(x) == False]
                bp_dict[one_task+"/"+metric_name] = metric_list

            bp_df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in bp_dict.items()]))
            plt_fig = plt.figure()
            a = bp_df.boxplot()
            plt.title(args.results_type+"_"+metric_name)
            plt_fig.savefig(str(args.performance_visulization_path)+"/"+args.results_type+"_{}.png".format(metric_name),format="png")
