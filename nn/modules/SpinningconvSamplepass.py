import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math
from torch.nn.init import constant_, xavier_uniform_
from ultralytics.utils.tal import TORCH_1_10, dist2bbox, make_anchors
from .block import DFL, Proto
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init_

# Define autopad function for padding calculation
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# Conv class implementation
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

# Provided C2f class
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super(C2f, self).__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)])

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend([m(y[-1]) for m in self.m])
        return self.cv2(torch.cat(y, 1))

# Provided Bottleneck class
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# Provided SpinningConv class
class SpinningConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super(SpinningConv, self).__init__()
        self.conv = Conv(c1, c2, k, s, p, g, d, act)

    def forward(self, x):
        angles = np.linspace(0, 360, 9)
        rotated_feature_maps = []

        for angle in angles:
            rotated_x = rotate(x, angle, interpolation=Image.BILINEAR)
            convolved_feature_map = self.conv(rotated_x)
            rotated_feature_maps.append(convolved_feature_map)

        sum_rotated_maps = torch.zeros_like(rotated_feature_maps[0])
        for feature_map in rotated_feature_maps:
            sum_rotated_maps += feature_map

        average_rotated_map = sum_rotated_maps / len(rotated_feature_maps)
        return average_rotated_map

def rotate(tensor, angle, interpolation=Image.BILINEAR):
    theta = torch.tensor([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
        [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0]
    ], dtype=torch.float)

    grid = F.affine_grid(theta.unsqueeze(0), tensor.size(), align_corners=False)
    return F.grid_sample(tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super(SPPF, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Upsample(nn.Module):
    """Upsample layer using nearest neighbor interpolation."""
    
    def __init__(self, scale_factor=None, mode='nearest'):
        """
        Initialize the Upsample layer with the specified scale factor and mode.

        Args:
            scale_factor (float or tuple): The scaling factor for upsampling. If a float, the same scaling factor
                                           is used for both height and width dimensions. If a tuple, the first value
                                           represents the scaling factor for the height dimension and the second value
                                           represents the scaling factor for the width dimension.
            mode (str): The upsampling algorithm to use. Default is 'nearest'.
        """
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """Perform forward pass through the Upsample layer."""
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
    
class Concat(nn.Module):
    """Concatenation layer."""

    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, inputs):
        """
        Concatenate input tensors along the channel dimension.

        Args:
            inputs (list): List of input tensors to concatenate.

        Returns:
            Tensor: Concatenated tensor.
        """
        return torch.cat(inputs, dim=1)

class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    
# Define the Backbone class
class Backbone(nn.Module):
    def __init__(self, nc=80):
        super(Backbone, self).__init__()
        self.layer0 = SpinningConv(3, 32, 3, 1)
        self.layer1 = SpinningConv(32, 32, 3, 1)
        self.layer2 = SpinningConv(32, 32, 3, 1)
        self.layer3 = C2f(32, 32, True)  # Assuming the correct implementation of C2f
        self.layer4 = Conv(32, 64, 3, 2)
        self.layer5 = Conv(64, 128, 3, 2)
        self.layer6 = C2f(128, 128, True)
        self.layer7 = Conv(128, 256, 3, 2)
        self.layer8 = C2f(256, 256, True)
        self.layer9 = Conv(256, 512, 3, 2)
        self.layer10 = C2f(512, 512, True)
        self.layer11 = Conv(512, 1024, 3, 2)
        self.layer12 = C2f(1024, 1024, True)
        self.layer13 = SPPF(1024, 512, 5)
        self.layer14 = Upsample(scale_factor=2, mode='nearest')
        self.layer15 = Concat()
        self.layer16 = C2f(1024, 1024, False)
        self.layer17 = Upsample(scale_factor=2, mode='nearest')
        self.layer18 = Concat()
        self.layer19 = C2f(1280, 1280, False)
        self.layer20 = Conv(1280,2560, 3, 2)
        self.layer21 = Concat()
        self.layer22 = C2f(3584, 3584, False)
        self.layer23 = Conv(3584,7168 , 3, 2)
        self.layer24 = Concat()
        self.layer25 = C2f(7680, 7680, False)
        ch = (1280, 3584, 7680)
        self.detect = Detect(nc,ch)

    
    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        x9 = self.layer9(x8)
        x10 = self.layer10(x9)
        x11 = self.layer11(x10)
        x12 = self.layer12(x11)
        x13 = self.layer13(x12)
        x14 = self.layer14(x13)
        x15 = self.layer15([x10, x14])  # Concatenating the outputs of the third and fourth layers
        x16 = self.layer16(x15)
        x17 = self.layer17(x16)
        x18 = self.layer18([x8, x17])
        x19 = self.layer19(x18)
        x20 = self.layer20(x19)
        x21 = self.layer21([x16, x20])
        x22 = self.layer22(x21)
        x23 = self.layer23(x22)
        x24 = self.layer24([x13,x23])
        x25 = self.layer25(x24)

        
        return self.detect([x19,x22,x25])


# Create the model
model = Backbone()



# Test with input tensor of shape (1, 3, 640, 480)
input_tensor = torch.randn(1, 3, 640, 480)
output_tensor = model(input_tensor)

print(f"Output shape: {output_tensor.shape}")
