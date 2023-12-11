# ControlNet的实现:
目前实现了3种网络结构, 写在./T2I-DIT-feat-pixart-ctrlnet/diffusion/model/nets/controlnet.py

**ControlPixArt_Mid**: 用前14个controlnet block 提feature，但只在中间层给控制

**ControlPixArtHalf**: 用前14层提condition的feature, 然后送到和这14个block对应的block里，比如说controlent.block[0]用base_block[0]的weight初始化, controlnet.block[0]的输出会和base_block[0]的输出一起送到base_block[1]

**ControlPixArtHalfRes1024**: ControlPixArtHalf的1024版本, 只修改了forward函数为了和pixartms.py的forward函数对应

**ControlPixArtAll**: 前14层的feature送到后14层里，类似controlnet+sd的结构。只support 1024

主要的改动:
1. 去掉了with torch.no_grad(), 靠optimizer的优化param来控制训练的param. 防止网络的反传有问题
2. condition = data_info['cond'] * scale_factor
3. condition + pos_embedding

# Training
新加的args
1. --resume_optimizer
2. --resume_lr_scheduler
3. --controlnet_type, 从 'all'和'half'里选

1&2 用来控制是否要resume之前的lr

# Inference:
**批量训练**

新写了一个inference函数 ./T2I-DIT-feat-pixart-ctrlnet/scripts/inference_ctrl_hed_train.py
这个函数用来批量测试图片. mode支持'train', 'inference'

默认使用dataset里前10%的图片用来训练
使用dataset里后10%的图片用来测试。 train模式会测试训练集的效果，inference模式会测试测试集的效果

结果会存在./output_demo/下面。会根据exp_id, step, 和mode创建路径

example shells are in ./test_inference_images.sh

**demo**

demo使用的是原函数./T2I-DIT-feat-pixart-ctrlnet/scripts/inference_ctrl_hed.py
example shells are in ./test.sh


# 实验记录
实验结果都存在/efs_cv/yue目录下面
network_all + res1024的实验
1. controlnet_all
2. controlnet_all_0.5data_drop0.3
3. controlnet_all_0.5data_drop0.3_lr2e-5
4. controlnet_all_0.5data_drop0.3_lr2e-5_fix
5. controlnet_all_0.5data_drop0.3_lr2e-5_fix_continue
每个实验都是从上一个实验finetune

network half + res 512
1. controlnet_half_0.1data_drop0.5_lr2e-5_res512

network half + res 1024
1. controlnet_half_0.1data_drop0.5_lr2e-5_autolr_fix_res1024

# 挂实验
run文件储存在 /home/xieenze/yue/s3helper/s3helper/scripts/run1024.sh

resume_from和resume_optimizer_component 这两个参数的解释在sh文件中
