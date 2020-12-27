import torch.nn as nn
import torch.nn.functional as F
from miscc.config import cfg
from utils.loss import cosine_similarity
import torch
from torch.autograd import Variable

# 1x1 convolution network 
def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)

# 3x3 convolution network with keep size
def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# GLU(Gated Linear Unit) module. It can be used like activation function
class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])

# ACM(affine combination module) in ManiGAN paper
class ACM(nn.Module): 
    def __init__(self, channel_num):
        super(ACM, self).__init__()
        self.conv = conv3x3(cfg.GAN.GF_DIM, 128)
        self.conv_weight = conv3x3(128, channel_num) 
        self.conv_bias = conv3x3(128, channel_num)      

    def forward(self, x, img):
        out_code = self.conv(img)
        out_code_weight = self.conv_weight(out_code)
        out_code_bias = self.conv_bias(out_code)
        return x * out_code_weight + out_code_bias


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.InstanceNorm2d(out_planes * 2),
        GLU())
        # conv3x3(in_planes, out_planes),
        # nn.InstanceNorm2d(out_planes),
        # nn.ReLU())
    return block
    

# Residual block in ManiGAN paper
class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.InstanceNorm2d(channel_num * 2),
            GLU(),
            # conv3x3(channel_num, channel_num),
            # nn.InstanceNorm2d(channel_num),
            # nn.ReLU(),
            conv3x3(channel_num, channel_num),
            nn.InstanceNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual # skip connection
        return out


def visual_semantic_loss(image, caps, dim=1, eps=1e-8):
    loss = -cosine_similarity(image,caps)
    loss = loss.mean()

    return loss

def pairwise_ranking_loss(x, v):
    margin = cfg.TRAIN.PRE_MARGIN
    zero = torch.zeros(1)
    diag_margin = margin * torch.eye(x.size(0))
    if cfg.CUDA:
        zero, diag_margin = zero.cuda(), diag_margin.cuda()
    zero, diag_margin = Variable(zero), Variable(diag_margin)

    x = x / torch.norm(x, 2, 1, keepdim=True)
    v = v / torch.norm(v, 2, 1, keepdim=True)
    prod = torch.matmul(x, v.transpose(0, 1))
    diag = torch.diag(prod)
    for_x = torch.max(zero, margin - torch.unsqueeze(diag, 1) + prod) - diag_margin
    for_v = torch.max(zero, margin - torch.unsqueeze(diag, 0) + prod) - diag_margin
    return (torch.sum(for_x) + torch.sum(for_v)) / x.size(0)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)