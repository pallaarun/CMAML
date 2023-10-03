import sys
import argparse
from torch.utils.data import DataLoader
from dataset import SliceData_validation

def create_data_loaders(args):

    data = SliceData_validation(args.data_path,args.degradation_amount,args.degradation_type)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )

    return data_loader

def main(args):
    
    data_loader = create_data_loaders(args)
    print("Task name =",args.degradation_type+"/"+args.degradation_amount)
    print("# of data points =",len(data_loader))

def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")

    parser.add_argument('--batch-size', default=1, type=int, help='Mini-batch size')
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

        args.data_path = args.data_dir+'/'+args.degradation_type+'/'+args.degradation_amount+'/'+args.flag
        main(args)
