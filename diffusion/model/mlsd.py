import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import cv2
import numpy as np
import torch
import json
from diffusers.models import AutoencoderKL
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import shutil

image_resize = 512


class BlockTypeA(nn.Module):
    def __init__(self, in_c1, in_c2, out_c1, out_c2, upscale=True):
        super(BlockTypeA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c2, out_c2, kernel_size=1),
            nn.BatchNorm2d(out_c2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c1, out_c1, kernel_size=1),
            nn.BatchNorm2d(out_c1),
            nn.ReLU(inplace=True)
        )
        self.upscale = upscale

    def forward(self, a, b):
        b = self.conv1(b)
        a = self.conv2(a)
        if self.upscale:
            b = F.interpolate(b, scale_factor=2.0, mode='bilinear', align_corners=True)
        return torch.cat((a, b), dim=1)


class BlockTypeB(nn.Module):
    def __init__(self, in_c, out_c):
        super(BlockTypeB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x)
        return x


class BlockTypeC(nn.Module):
    def __init__(self, in_c, out_c):
        super(BlockTypeC, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        self.channel_pad = out_planes - in_planes
        self.stride = stride
        # padding = (kernel_size - 1) // 2

        # TFLite uses slightly different padding than PyTorch
        if stride == 2:
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        # TFLite uses  different padding
        if self.stride == 2:
            x = F.pad(x, (0, 1, 0, 1), "constant", 0)
            # print(x.shape)

        for module in self:
            if not isinstance(module, nn.MaxPool2d):
                x = module(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=True):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        width_mult = 1.0
        round_nearest = 8

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            # [6, 160, 3, 2],
            # [6, 320, 1, 1],
        ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(4, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.features = nn.Sequential(*features)
        self.fpn_selected = [1, 3, 6, 10, 13]
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        if pretrained:
            self._load_pretrained_model()

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        fpn_features = []
        for i, f in enumerate(self.features):
            if i > self.fpn_selected[-1]:
                break
            x = f(x)
            if i in self.fpn_selected:
                fpn_features.append(x)

        c1, c2, c3, c4, c5 = fpn_features
        return c1, c2, c3, c4, c5

    def forward(self, x):
        return self._forward_impl(x)

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def deccode_output_score_and_ptss(tpMap, topk_n=200, ksize=5):
    '''
    tpMap:
    center: tpMap[1, 0, :, :]
    displacement: tpMap[1, 1:5, :, :]
    '''
    c, h, w = tpMap.shape
    displacement = tpMap[1:5, :, :]
    center = tpMap[[0], :, :]
    heat = torch.sigmoid(center)
    hmax = F.max_pool2d(heat, (ksize, ksize), stride=1, padding=(ksize - 1) // 2)
    keep = (hmax == heat).float()
    heat = heat * keep
    heat = heat.reshape(-1, )

    scores, indices = torch.topk(heat, topk_n, dim=-1, largest=True)
    yy = torch.floor_divide(indices, w).unsqueeze(-1)
    xx = torch.fmod(indices, w).unsqueeze(-1)
    ptss = torch.cat((yy, xx), dim=-1)

    ptss = ptss.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    displacement = displacement.detach().cpu().numpy()
    displacement = displacement.transpose((1, 2, 0))
    return ptss, scores, displacement


def pred_lines(model: torch.nn.Module, batch_image: torch.Tensor,
               score_thr=0.10,
               dist_thr=20.0):
    outputs = model(batch_image)
    lines = []
    for output in outputs:
        pts, pts_score, vmap = deccode_output_score_and_ptss(output, 200, 3)
        start = vmap[:, :, :2]
        end = vmap[:, :, 2:]
        dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

        segments_list = []
        for center, score in zip(pts, pts_score):
            y, x = center
            distance = dist_map[y, x]
            if score > score_thr and distance > dist_thr:
                disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
                x_start = x + disp_x_start
                y_start = y + disp_y_start
                x_end = x + disp_x_end
                y_end = y + disp_y_end
                segments_list.append([x_start, y_start, x_end, y_end])

        lines.append(2 * np.array(segments_list))  # 256 > 512
    return lines


class MobileV2_MLSD_Large(nn.Module):
    def __init__(self):
        super(MobileV2_MLSD_Large, self).__init__()

        self.backbone = MobileNetV2(pretrained=False)
        ## A, B
        self.block15 = BlockTypeA(in_c1=64, in_c2=96,
                                  out_c1=64, out_c2=64,
                                  upscale=False)
        self.block16 = BlockTypeB(128, 64)

        ## A, B
        self.block17 = BlockTypeA(in_c1=32, in_c2=64,
                                  out_c1=64, out_c2=64)
        self.block18 = BlockTypeB(128, 64)

        ## A, B
        self.block19 = BlockTypeA(in_c1=24, in_c2=64,
                                  out_c1=64, out_c2=64)
        self.block20 = BlockTypeB(128, 64)

        ## A, B, C
        self.block21 = BlockTypeA(in_c1=16, in_c2=64,
                                  out_c1=64, out_c2=64)
        self.block22 = BlockTypeB(128, 64)

        self.block23 = BlockTypeC(64, 16)

    def forward(self, x):
        c1, c2, c3, c4, c5 = self.backbone(x)

        x = self.block15(c4, c5)
        x = self.block16(x)

        x = self.block17(c3, x)
        x = self.block18(x)

        x = self.block19(c2, x)
        x = self.block20(x)

        x = self.block21(c1, x)
        x = self.block22(x)
        x = self.block23(x)
        x = x[:, 7:, :, :]

        return x


class MJData(Dataset):
    def __init__(self):
        with open('/home/xieenze/ai-theory-enze-efs2/jincheng/mj_1_new.json', 'r+') as f:
            self.j = json.load(f)
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGBA')),
            T.Resize(image_resize),  # Image.BICUBIC
            T.CenterCrop(image_resize),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2 - 1.),
        ])

    def __len__(self):
        return len(self.j)

    def __getitem__(self, idx):
        path = self.j[idx]['path']
        image = Image.open("/home/xieenze/ai-theory-enze-efs2/jincheng/imgs/" + path)
        image = self.transform(image)
        return image, path


class MLSDdetector(nn.Module):
    def __init__(self):
        super().__init__()
        model = MobileV2_MLSD_Large()
        model.load_state_dict(torch.load("output/pretrained_models/mlsd_large_512_fp32.pth"), strict=True)
        self.model = model

    def forward(self, input_image):
        img = input_image
        lines = pred_lines(self.model, img, 0.1, 0.1)
        output = []
        for l in lines:
            img_output = np.zeros((image_resize, image_resize, 3), dtype=np.uint8)
            for line in l:
                x_start, y_start, x_end, y_end = [int(val) for val in line]
                cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 2)
            output.append(255 - img_output)
        return output


def extract(rank):
    with open('/home/xieenze/ai-theory-enze-efs2/jincheng/mj_1_new.json', 'r+') as f:
        j = json.load(f)
    clip = 3312260 // 32

    vae = AutoencoderKL.from_pretrained("output/pretrained_models/sd-vae-ft-ema")
    msld = MLSDdetector()
    device = f'cuda:{rank % 4}'
    vae.to(device)
    msld.model.to(device)
    image_resize = 512
    transform1 = T.Compose([
        T.Lambda(lambda img: img.convert('RGBA')),
        T.Resize(image_resize),  # Image.BICUBIC
        T.CenterCrop(image_resize),
        T.ToTensor(),
        T.Lambda(lambda x: x * 2 - 1.),
    ])
    transform2 = T.Compose([
        T.ToTensor(),
        T.Normalize([.5], [.5]),
    ])

    for i in tqdm(j[clip * rank: clip * (rank + 1)] if rank != 31 else j[clip * rank:], position=rank):
        path = "/home/xieenze/ai-theory-enze-efs2/jincheng/imgs/" + i['path']
        img = transform1(Image.open(path))
        with torch.inference_mode():
            img = Image.fromarray(msld(np.array(img), 0.1, 0.1, device))
            img = transform2(img).to(device)[None]
            posterior = vae.encode(img).latent_dist
            z = torch.cat([posterior.mean, posterior.std], dim=1).detach().cpu().numpy()
        save = '/home/xieenze/nfs_han/jincheng/mj_mlsd' + i['path'].replace('.png', '.npz')
        os.makedirs(os.path.dirname(save), exist_ok=True)
        np.savez_compressed(save, z)


with torch.inference_mode():
    dataset = MJData()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=1, pin_memory=True)
    mlsd = MLSDdetector()
    for img, path in dataloader:
        out = mlsd(img)
        for i, p, o in zip(img, path, out):
            shutil.copy('/home/xieenze/ai-theory-enze-efs2/jincheng/imgs/' + p, '/tmp/jincheng/' + p.replace('/', '_'))
            print(o.shape)
            Image.fromarray(o).convert('L').save('/tmp/jincheng/' + p.replace('/', '_') + '.mlsd.png')
        break
