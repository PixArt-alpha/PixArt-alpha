import os,sys,shutil
import subprocess
import os
import argparse
import random
from cloud_tools.cloud.utils.dist_utils import synchronize_ip

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument("config", type=str, help="config")
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from', help='the dir to save logs and models')
    parser.add_argument('--model_name', help='load test checkpoint')
    parser.add_argument('--np', type=int, help="nproc_per_node, number of GPUs per node")  # specify when mode is multiple
    parser.add_argument("--test", action='store_true', default=False, help="if use test mode")
    parser.add_argument('--run_script', type=str, default='train_scripts/train_controlnet.py')
    # for multiple machines
    parser.add_argument('--main_port', type=str, default='20480', help='Port of the current rank 0.')
    parser.add_argument('--machine_num', type=int, default=1, help='number of machine')
    parser.add_argument('--vis_gpu', type=str, default='0,1,2,3,4,5,6,7', help='visible gpu id')
    parser.add_argument('--bucket', type=str, default='cneast3')
    parser.add_argument('--controlnet_type', type=str, default='all')
    parser.add_argument('--resume_optimizer_component', type=str, default='optim_lr', help='choose from optim_lr or None. \
        load optimizer and lr schedule at the same time or None: only load the checkpoint')
    

    args = parser.parse_args()

    return args

args = parse_args()
gpu_num = args.np
vis_gpu = args.vis_gpu
run_dir = args.work_dir
print("saving dir: ", run_dir)

# config working dir
WORK_DIR = os.getcwd()
print("current working dir: ", WORK_DIR)
sys.path.insert(0, os.getcwd())

# environment variables
main_port = args.main_port
bucket = f"bucket-{args.bucket}"
user_name = 'yue'
s3_work_dir = f's3://{bucket}/{user_name}/code/console/ip_temp'
machine_num = args.machine_num
master_addr, machine_rank, host_ip = synchronize_ip(s3_work_dir, machine_num)
if machine_num > 1:
    master_addr, master_port = master_addr.split(':')
else:
    master_port = main_port

# launch script
run_script = args.run_script

print('################################################# We have install now! #########################################')
if not args.test:
    print('################################################# Start training! #########################################')
    run_cmd = 'python -m torch.distributed.launch ' \
              '--nproc_per_node={} ' \
              '--master_port={} ' \
              '--nnodes={} ' \
              '--master_addr={} ' \
              '--node_rank={} {} ' \
              '{} --cloud ' \
              '--work-dir {} '.format(gpu_num,
                                      master_port,
                                      machine_num,
                                      master_addr,
                                      machine_rank,
                                      run_script,
                                      args.config,
                                      run_dir)
    # 'scripts/train_imgnet.py',
    # 'scripts/train.py',
    if args.resume_from:
        run_cmd += '--resume_from {} '.format(args.resume_from)
    if args.controlnet_type:
        run_cmd += '--controlnet_type {} '.format(args.controlnet_type)
    if args.resume_optimizer_component == 'optim_lr':
        run_cmd += '--resume_optimizer --resume_lr_scheduler'
    # 'python -m torch.distributed.launch --nproc_per_node=8 --master_port=20004 --master_addr= --node_rank=0 scripts/train.py  --cloud --work_dir '

else:
    print('################################################# Start testing! #########################################')
    run_cmd = ''

    print(run_cmd)
    subprocess.call(run_cmd, shell=True)

print(run_cmd)
subprocess.call(run_cmd, shell=True)
