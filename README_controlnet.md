# ControlNet的实现:
目前实现了3种网络结构, 写在./T2I-DIT-feat-pixart-ctrlnet/diffusion/model/nets/controlnet.py

ControlPixArt_Mid: 用前14个controlnet block 提feature，但只在中间层给控制
ControlPixArtHalf: 用前14层提condition的feature, 然后送到和这14个block对应的block里，比如说controlent.block[0]用base_block[0]的weight初始化, controlnet.block[0]的输出会和base_block[0]的输出一起送到base_block[1]
只support 512res，因为需要从PixArt.py里抄forward函数。而PixArt.py和PixArtMS.py的forward函数有细微差别。
如果需要support 1024 res, 则需要从PixArtMS.py里抄再小改一下即可。
ControlPixArtAll: 前14层的feature送到后14层里，类似controlnet+sd的结构。只support 1024



