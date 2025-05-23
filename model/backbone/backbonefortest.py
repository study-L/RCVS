import torch.nn.functional as F
import pywt
import copy
import torch
import torch.nn as nn
from torch.nn import Module
import math
from einops.einops import rearrange
import torch
from torch import nn
from timm.models.layers import DropPath, trunc_normal_
from torch.autograd import Function
# from mmcv.cnn import get_model_complexity_info
import matplotlib.pyplot as plt
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
# class CL(nn.Module):
#     def __init__(self, in_planes, planes):
#         super().__init__()
#         self.conv = conv3x3(in_planes, planes)
#         self.bn = nn.BatchNorm2d(planes)
#         self.relu =nn.LeakyReLU(inplace=True)
#
#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))
class CL(nn.Module):
    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv = conv3x3(in_planes, planes)
        self.bn = nn.BatchNorm2d(planes)
        self.relu =nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu( self.bn(self.conv(x)))
class TCBR(nn.Module):
    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_planes, planes, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,groups=in_planes, padding=1, bias=False)

class CBD0(nn.Module):
    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
class CBR(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv = conv3x3(in_planes, planes, stride)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class JCBR(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, stride=2,padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
class CBD(nn.Module):
    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv = conv3x3(in_planes, planes, stride=2)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
class Fusion_AVG(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = (en_ir + en_vi) / 2
        return temp
class MLP2(nn.Module):
    def __init__(self, in_features, mlp_ratio=4):
        super(MLP2, self).__init__()
        hidden_features = in_features * mlp_ratio

        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, in_features)
        )

    def forward(self, x):
        return self.fc(x)
class CrossModalAttention(nn.Module):
    """Cross-modal semantic calibration

    Args:
        'feat' (torch.Tensor): (N, L)
        'attention' (torch.Tensor): (N, L, C)
    """

    def __init__(self, dim, mlp_ratio=2, qkv_bias=False):
        super(CrossModalAttention, self).__init__()
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP2(dim, mlp_ratio)

    def forward(self, feat, attention):
        shortcut = feat
        feat = self.norm1(feat)
        feat = self.qkv(feat)
        x = torch.einsum('nl, nlc -> nlc', attention, feat)
        x = self.proj_out(x)
        x = x + shortcut
        x = x + self.mlp(self.norm2(x))
        return x
class Feature_Extraction(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self):
        super().__init__()
        # Config
        initial_dim = 8

        # Class Variable
        self.in_planes = initial_dim

        self.dwt = DWT_2D(wave='haar')
        # Networks
        # 1/2
        # self.conv0 =  nn.Conv2d(1, 256, kernel_size=8, stride=8)
        self.conv0 = nn.Conv2d(1, 256, kernel_size=8, stride=8)
        self.norm = nn.BatchNorm2d(256)
        self.norm2 = nn.LayerNorm(256,eps=1e-6)
        self.norm3 = nn.LayerNorm(256, eps=1e-6)

        self.body = nn.Sequential(JConv(256, 256),JConv(256, 256),JConv(256, 256),JConv(256, 256))

        self.pos_encoding = nn.Sequential(conv3x3(256,256),nn.Sigmoid())
        # self.transattention_vi1=JConv(256, 256)
        # self.transattention_vi2 = StructureAttention(256,8)
        #
        # self.transattention_ir1 = JConv(256, 256)
        # self.transattention_ir2 = StructureAttention(256,8)
        # self.transattention_vi1 = JConv(256, 256)
        self.transattention_vi2 = Block(256, 2, cross=True)
        # self.transattention_vi3 = StructureAttention(256, 8)

        # self.transattention_ir1 = JConv(256, 256)
        # self.transattention_ir2 = Block(256, 8, Cross=True)
        # # self.transattention_ir3 = StructureAttention(256, 8)









        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):



        x0 = self.conv0(x)
        x0=self.norm(x0)
        x0= self.body(x0)




        # (feat_c2, feat_f2) = x0.split(bs)
        # # feat_c2, feat_f2 = self.backbone(img2)
        # plt.figure(figsize=(2, 1))
        # layer_viz = feat_c2[0, :, :, :].detach().cpu()
        # layer_viz = layer_viz.data
        # for i, filter in enumerate(layer_viz):
        #     if i == 256:
        #         break
        #     plt.subplot(16, 16, i + 1)
        #     plt.imshow(filter,cmap='jet')  # 如果需要彩色的，可以修改cmap的参数
        #     plt.axis("off")
        # plt.savefig(r'E:\LDC-main\checkpoints\vi-hei.png',dpi=10000)
        #
        # plt.figure(figsize=(2, 1))
        # layer_viz = feat_f2[0, :, :, :].detach().cpu()
        # layer_viz = layer_viz.data
        # for i, filter in enumerate(layer_viz):
        #     if i == 256:
        #         break
        #     plt.subplot(16, 16, i + 1)
        #     plt.imshow(filter, cmap='jet')  # 如果需要彩色的，可以修改cmap的参数
        #     plt.axis("off")
        # plt.savefig(r'E:\LDC-main\checkpoints\ir-hei.png', dpi=10000)

        # x0=self.norm(x0)


        bs = int(x0.shape[0] / 2)


        # feat_0 = self.transattention_vi1(feat_00)
        # feat_1 = self.transattention_ir1(feat_11)

        h = x0.shape[2]
        w = x0.shape[3]
        #
        feat=self.pos_encoding(x0)*x0
        # feat= self.norm2(feat)
        # feat_1 = self.pos_encoding(feat_11) * feat_11
        #
        feat = rearrange(feat, 'n c h w -> n (h w) c')

        #
        feat = self.transattention_vi2(feat,h,w)
        feat = self.norm2(feat)

        # feat_1 = self.transattention_ir2(feat_1, feat_0)
        #
        # # feat_0 = self.transattention_vi3(feat_0, feat_1)
        # # feat_1 = self.transattention_ir3(feat_1, feat_0)
        #
        # # (feat_00, feat_11) = x0.split(bs)
        # # feat_0=self.transattention_vi1(feat_00)
        # # feat_1 = self.transattention_ir1(feat_11)
        # h=feat_0.shape[2]
        # w = feat_0.shape[3]
        # feat_0 = rearrange(feat_0, 'n c h w -> n (h w) c')
        # feat_1 = rearrange(feat_1, 'n c h w -> n (h w) c')
        # #
        # # feat_0=self.transattention_vi2(feat_0,feat_0)
        # feat_1 = self.transattention_ir2(feat_1,feat_1)
        #
        feat = rearrange(feat, 'n (h w) c -> n c h w',h=h,w=w).contiguous()
        # feat_1 = rearrange(feat_1, 'n (h w) c -> n c h w',h=h,w=w)
        #
        feat=feat*x0



        # feat_0 = rearrange(feat_0, 'n (h w) c -> n c h w', h=h, w=w)
        # feat_1 = rearrange(feat_1, 'n (h w) c -> n c h w', h=h, w=w)

        # feat_0 = feat_0 * feat_00
        # feat_1 = feat_1 * feat_11

        return feat










        # edge = self.edge(x)


        # bs = int(x.shape[0]/2)
        #
        # (feat_0, feat_1) = x.split(bs)
        # h=feat_0.shape[2]
        # w = feat_0.shape[3]
        #
        # feat_0 = rearrange(feat_0, 'n c h w -> n (h w) c')
        # feat_1 = rearrange(feat_1, 'n c h w -> n (h w) c')
        # conf = torch.einsum("nlc,nsc->nls", feat_0,
        #                             feat_1)/0.1
        # conf = F.softmax(conf, 1) * F.softmax(conf, 2)
        # conf_x=torch.max(conf,dim=1)[0]
        # conf_y = torch.max(conf, dim=2)[0]
        #
        # feat_h = h
        # feat_w = w
        # # Predefined spatial grid
        # xs = torch.linspace(0, feat_h - 1, feat_h)
        # ys = torch.linspace(0, feat_w - 1, feat_w)
        # xs = xs / (feat_h - 1)
        # ys = ys / (feat_w - 1)
        # grid = torch.stack(torch.meshgrid([xs, ys]), dim=-1).unsqueeze(0).repeat(int(feat_0.shape[0]), 1, 1,
        #                                                                          1)
        # grid=rearrange(grid, 'n h w c-> n c h w',h=h,w=w)
        #
        # # conf_x = rearrange(conf_x, 'n (h w) -> n h w',h=h,w=w)
        # # conf_y = rearrange(conf_y, 'n (h w) -> n h w', h=h, w=w)
        #
        # grid=self.pos_encoder(grid)
        # grid = rearrange(grid, 'n c h w-> n (h w) c', h=h, w=w)
        # # gridy = rearrange(gridy, 'n c h w-> n (h w) c', h=h, w=w)
        # grid_0x,grid1x=torch.split(grid, [256, 256],dim=2)
        # grid_0y,grid1y=grid_0x,grid1x
        #
        #
        #
        # grid_0x,grid0y=self.att(grid_0x,conf_x),self.att(grid_0y,conf_y)
        # gridx,gridy=self.attention(grid1x,grid_0x),self.attention(grid1y,grid_0y)
        # conf = torch.einsum("nlc,nsc->nls", gridx,
        #                     gridy) / 0.1
        # conf = F.softmax(conf, 1) * F.softmax(conf, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 cross=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.cross = cross

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.cross == True:
            MiniB = B // 2
            # cross attention
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q1, q2 = q.split(MiniB)

            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            k1, k2 = kv[0].split(MiniB)
            v1, v2 = kv[1].split(MiniB)

            attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)

            attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)

            x1 = (attn1 @ v2).transpose(1, 2).reshape(MiniB, N, C)
            x2 = (attn2 @ v1).transpose(1, 2).reshape(MiniB, N, C)

            x = torch.cat([x1, x2], dim=0)
        else:
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=4, cross = False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, cross= cross)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)

        return x
# class ResNetFPN_8_2(nn.Module):
#     """
#     ResNet+FPN, output resolution are 1/8 and 1/2.
#     Each block has 2 layers.
#     """
#
#     def __init__(self):
#         super().__init__()
#         # Config
#         initial_dim = 8
#
#         # Class Variable
#         self.in_planes = initial_dim
#
#         self.dwt = DWT_2D(wave='haar')
#         # Networks
#         # 1/2
#         self.conv1 = JConv(1, 8)
#         # self.gct1=GCT(32)
#         self.conv2 = JConv(32, 16)
#         # self.gct2= GCT(64)
#         self.conv3 = JConv(64, 32)
#         self.conv4 = JConv(256, 256)
#         # self.gct3 = GCT(256)
#         # self.body = nn.Sequential(NCB(256, 256), NCB(256, 256), NCB(256, 256))
#         #
#         # self.pred1 = nn.Sequential(NCB(256, 256), NCB(256, 256))
#         #
#         # # self.pred1 = NCB(256, 256)
#         # self.pred2 = NCB(256, 1)
#         # self.pred3 = nn.Sigmoid()
#
#         self.body = nn.Sequential(JConv(256, 256), JConv(256, 256), JConv(256, 256))
#
#         self.pred1 = JConv(256, 256)
#         self.pred2 = JConv(256, 256)
#         self.pred3 = JConv(256, 64)
#         self.pred4 = JConv(64, 1)
#         self.pred5 = nn.Sigmoid()
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         # ResNet Backbone
#         x = self.dwt(self.conv1(x))  # 1/2
#         x = self.dwt(self.conv2(x))  # 1/4
#         x = self.conv4(self.dwt(self.conv3(x)))
#         x = self.body(x)
#
#         pred = self.pred1(x)
#         pred = self.pred2(pred)
#         pred = self.pred3(pred)
#         pred = self.pred4(pred)
#         pred = self.pred5(pred)
#         return x,pred
#


#
# class ResNetFPN_8_2(nn.Module):
#     """
#     ResNet+FPN, output resolution are 1/8 and 1/2.
# #     Each block has 2 layers.
# #     """
#
#     def __init__(self):
#         super().__init__()
#         # Config
#         initial_dim = 8
#
#         # Class Variable
#         self.in_planes = initial_dim
#
#         self.dwt = DWT_2D(wave='haar')
#         # Networks
#         # 1/2
#         self.conv1 = JConv(1, 8)
# #         # self.gct1=GCT(32)
#         self.conv2 = JConv(32, 16)
#         # self.gct2= GCT(64)
#         self.conv3 = JConv(64, 32)
#         self.conv4 = JConv(256, 256)
#         # self.gct3 = GCT(256)
#         # self.body = nn.Sequential(NCB(256, 256), NCB(256, 256), NCB(256, 256))
#         #
#         # self.pred1 = nn.Sequential(NCB(256, 256), NCB(256, 256))
#         #
#         # # self.pred1 = NCB(256, 256)
#         # self.pred2 = NCB(256, 1)
#         # self.pred3 = nn.Sigmoid()
#
#         self.body = nn.Sequential(JConv(256, 256), JConv(256, 256), JConv(256, 256))
#
#         self.pred1 = JConv(256, 256)
#         self.pred2 = JConv(256, 256)
#         self.pred3 = JConv(256, 64)
#         self.pred4 = JConv(64, 1)
#         self.pred5 = nn.Sigmoid()
#
#         self.cross1 = Cross(256)
#         self.cross2 = Cross(256)
#         # self.cross2 = Cross(256)
#         # self.cross3 = Cross(256)
#         # self.cross4 = Cross(256)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         # ResNet Backbone
#         x = self.dwt(self.conv1(x))  # 1/2
#         x = self.dwt(self.conv2(x))  # 1/4
#         x = self.conv4(self.dwt(self.conv3(x)))
#         x = self.body(x)
#
#         bs2 = x.shape[0]
#         (featmask, featmaskir) = x.split(int(bs2 / 2))
#         h = featmask.shape[2]
#         w = featmask.shape[3]
#
#         featmaskirsigmoid = self.pred1(featmaskir)
#         featmaskirsigmoid = self.pred2(featmaskirsigmoid)
#         featmaskirsigmoid = self.pred3(featmaskirsigmoid)
#         featmaskirsigmoid = self.pred4(featmaskirsigmoid)
#         featmaskirsigmoid = self.pred5(featmaskirsigmoid)
#         featmaskirsigmoidx = rearrange(featmaskirsigmoid, 'n c h w -> n c (h w)')
#
#         featmask = rearrange(featmask, 'n c h w -> n (h w) c')
#         featmaskir = rearrange(featmaskir, 'n c h w -> n (h w) c')
#
#         feat1, feat2 = map(lambda feat: feat / feat.shape[-1] ** .5,
#                            [featmask, featmaskir])
#         attention = torch.einsum("nlc,nsc->nls", feat1,
#                                  feat2) / 0.1
#
#         attention = attention.softmax(dim=1)
#
#
#         attention = torch.einsum("nls,ncs->nls", attention,
#                                  featmaskirsigmoidx)
#
#         # attention = attention.softmax(dim=-2)
#
#         attention = torch.sum(attention, dim=2)
#
#         # att = torch.clamp(attention, 0, 1)
#
#         featmask = self.cross1(featmask, attention*2)
#         featmask = self.cross2(featmask, attention*2)
#         # featmask = self.cross2(featmask, attention)
#         # featmask = self.cross3(featmask, attention)
#         #
#         # attention2 = torch.einsum("nlc,nsc->nls", feat2,
#         #                           feat1) / 0.1
#         # attention2 = attention2.softmax(dim=-1)
#         # featmask=self.cross1(featmask,atten
#         # featmaskir = self.cross2(featmaskir, attention2)
#         # featmask = self.cross3(featmask, attention1)
#         # featmaskir = self.cross4(featmaskir, attention2)
#         featmask = rearrange(featmask, 'n (h w) c -> n c h w', h=h, w=w)
#         # featmaskir = rearrange(featmaskir, 'n (h w) c -> n c h w', h=h, w=w)
#
#         # featmask=torch.einsum('nls, nlc -> nlc', attention, v)
#
#         featmask = self.pred1(featmask)
#         featmask = self.pred2(featmask)
#         featmask = self.pred3(featmask)
#         featmask = self.pred4(featmask)
#         featmasksigmoid = self.pred5(featmask)
#
#         # featmaskir = self.pred1(featmaskir)
#         # featmaskir=self.pred2(featmaskir)
#         # featmaskirsigmoid = self.pred3(featmaskir)
#
#         # pred = self.pred1(x)
#         # pred = self.pred2(pred)
#         # pred = self.pred3(pred)
#         # bs2 = x.shape[0]
#         # (featmasksigmoid, featmaskirsigmoid) = pred.split(int(bs2 / 2))
#
#         #
#         # featmask,featmaskir = rearrange(featmask, 'n c h w -> n (h w) c'),rearrange(featmaskir, 'n c h w -> n (h w) c')
#         # featmask = rearrange(self.pos_encoding(featmask), 'n c h w -> n (h w) c')
#         # featmaskir = rearrange(self.pos_encoding(featmaskir), 'n c h w -> n (h w) c')
#         # pred = self.pred2(pred)
#         # pivi, piir = x.split(int(bs2 / 2))
#         # gridvi, gridir = self.shape(featmasksigmoid, featmaskirsigmoid)
#         # piir = gridir + piir
#         # pivi = gridvi + pivi
#         # return torch.cat((pivi, piir), dim=0),  torch.cat((featmasksigmoid, featmaskirsigmoid), dim=0)
#         # pred = self.pred3(pred)  # 1/8
#         return x, torch.cat((featmasksigmoid, featmaskirsigmoid), dim=0)
#
#
#
# class ResNetFPN_8_2(nn.Module):
#     """
#     ResNet+FPN, output resolution are 1/8 and 1/2.
# #     Each block has 2 layers.
# #     """
#
#     def __init__(self):
#         super().__init__()
#         # Config
#         initial_dim = 8
#
#         # Class Variable
#         self.in_planes = initial_dim
#
#         self.dwt = DWT_2D(wave='haar')
#         # Networks
#         # 1/2
#         self.conv1 = JConv(1, 8)
# #         # self.gct1=GCT(32)
#         self.conv2 = JConv(32, 16)
#         # self.gct2= GCT(64)
#         self.conv3 = JConv(64, 32)
#         self.conv4 = JConv(256, 256)
#         # self.gct3 = GCT(256)
#         # self.body = nn.Sequential(NCB(256, 256), NCB(256, 256), NCB(256, 256))
#         #
#         # self.pred1 = nn.Sequential(NCB(256, 256), NCB(256, 256))
#         #
#         # # self.pred1 = NCB(256, 256)
#         # self.pred2 = NCB(256, 1)
#         # self.pred3 = nn.Sigmoid()
#
#         self.body = nn.Sequential(JConv(256, 256), JConv(256, 256), JConv(256, 256))
#
#         self.pred1 = JConv(256, 256)
#         self.pred2 = JConv(256, 256)
#         self.pred3 = JConv(256, 64)
#         self.pred4 = JConv(64, 1)
#         self.pred5 = nn.Sigmoid()
#
#         self.cross1 = Cross(256)
#         self.cross2 = Cross(256)
#         self.shape=Shape()
#         # self.cross2 = Cross(256)
#         # self.cross3 = Cross(256)
#         # self.cross4 = Cross(256)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         # ResNet Backbone
#         x = self.dwt(self.conv1(x))  # 1/2
#         x = self.dwt(self.conv2(x))  # 1/4
#         x = self.conv4(self.dwt(self.conv3(x)))
#         x = self.body(x)
#
#         bs2 = x.shape[0]
#         (featmask, featmaskir) = x.split(int(bs2 / 2))
#         h = featmask.shape[2]
#         w = featmask.shape[3]
#
#         featmaskirsigmoid = self.pred1(featmaskir)
#         featmaskirsigmoid = self.pred2(featmaskirsigmoid)
#         featmaskirsigmoid = self.pred3(featmaskirsigmoid)
#         featmaskirsigmoid = self.pred4(featmaskirsigmoid)
#         featmaskirsigmoid = self.pred5(featmaskirsigmoid)
#         featmaskirsigmoidx = rearrange(featmaskirsigmoid, 'n c h w -> n c (h w)')
#
#         featmask = rearrange(featmask, 'n c h w -> n (h w) c')
#         featmaskir = rearrange(featmaskir, 'n c h w -> n (h w) c')
#
#         feat1, feat2 = map(lambda feat: feat / feat.shape[-1] ** .5,
#                            [featmask, featmaskir])
#         attention = torch.einsum("nlc,nsc->nls", feat1,
#                                  feat2) / 0.1
#
#         attention = attention.softmax(dim=1)
#
#
#         attention = torch.einsum("nls,ncs->nls", attention,
#                                  featmaskirsigmoidx)
#
#         # attention = attention.softmax(dim=-2)
#
#         attention = torch.sum(attention, dim=2)
#
#         # att = torch.clamp(attention, 0, 1)
#
#         featmask = self.cross1(featmask, attention*1.5)
#         featmask = self.cross2(featmask, attention*1.5)
#         # featmask = self.cross2(featmask, attention)
#         # featmask = self.cross3(featmask, attention)
#         #
#         # attention2 = torch.einsum("nlc,nsc->nls", feat2,
#         #                           feat1) / 0.1
#         # attention2 = attention2.softmax(dim=-1)
#         # featmask=self.cross1(featmask,atten
#         # featmaskir = self.cross2(featmaskir, attention2)
#         # featmask = self.cross3(featmask, attention1)
#         # featmaskir = self.cross4(featmaskir, attention2)
#         featmask = rearrange(featmask, 'n (h w) c -> n c h w', h=h, w=w)
#         # featmaskir = rearrange(featmaskir, 'n (h w) c -> n c h w', h=h, w=w)
#
#         # featmask=torch.einsum('nls, nlc -> nlc', attention, v)
#
#         featmask = self.pred1(featmask)
#         featmask = self.pred2(featmask)
#         featmask = self.pred3(featmask)
#         featmask = self.pred4(featmask)
#         featmasksigmoid = self.pred5(featmask)
#
#         # featmaskir = self.pred1(featmaskir)
#         # featmaskir=self.pred2(featmaskir)
#         # featmaskirsigmoid = self.pred3(featmaskir)
#
#         # pred = self.pred1(x)
#         # pred = self.pred2(pred)
#         # pred = self.pred3(pred)
#         # bs2 = x.shape[0]
#         # (featmasksigmoid, featmaskirsigmoid) = pred.split(int(bs2 / 2))
#
#         #
#         # featmask,featmaskir = rearrange(featmask, 'n c h w -> n (h w) c'),rearrange(featmaskir, 'n c h w -> n (h w) c')
#         # featmask = rearrange(self.pos_encoding(featmask), 'n c h w -> n (h w) c')
#         # featmaskir = rearrange(self.pos_encoding(featmaskir), 'n c h w -> n (h w) c')
#         # pred = self.pred2(pred)
#         pivi, piir = x.split(int(bs2 / 2))
#         gridvi, gridir = self.shape(featmasksigmoid, featmaskirsigmoid)
#         piir = gridir + piir
#         pivi = gridvi + pivi
#         return torch.cat((pivi, piir), dim=0),  torch.cat((featmasksigmoid, featmaskirsigmoid), dim=0)
#         # pred = self.pred3(pred)  # 1/8
#         # return x, torch.cat((featmasksigmoid, featmaskirsigmoid), dim=0)
#

class BaseBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=12, ratios=[2, 2], qkv_bias=False):
        super(BaseBlock, self).__init__()
        self.layers = nn.ModuleList([])
        for ratio in ratios:
            self.layers.append(nn.ModuleList([
                Transformer(dim, num_heads, window_size, ratio, qkv_bias),
                ResBlock(dim, ratio)
            ]))

    def forward(self, x):
        for tblock, rblock in self.layers:
            x = tblock(x)
            x = rblock(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, mlp_ratio=4, qkv_bias=False):
        super(Transformer, self).__init__()
        self.window_size = window_size
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttn(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = rearrange(x, 'b c h w -> b h w c')
        b, h, w, c = x.shape

        shortcut = x
        x = self.norm1(x)

        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, c
        x_windows = rearrange(x_windows, 'B s1 s2 c -> B (s1 s2) c', s1=self.window_size,
                              s2=self.window_size)  # nW*b, window_size*window_size, c

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*b, window_size*window_size, c

        # merge windows
        attn_windows = rearrange(attn_windows, 'B (s1 s2) c -> B s1 s2 c', s1=self.window_size, s2=self.window_size)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # b H' W' c

        # reverse cyclic shift
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x + shortcut
        x = x + self.mlp(self.norm2(x))
        return rearrange(x, 'b h w c -> b c h w')


class ResBlock(nn.Module):
    def __init__(self, in_features, ratio=2):
        super(ResBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_features, in_features * ratio, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_features * ratio, in_features * ratio, 3, 1, 1, groups=in_features * ratio),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_features * ratio, in_features, 1, 1, 0),
        )

    def forward(self, x):
        return self.net(x) + x


# def elu_feature_map(x):
#     return torch.nn.functional.elu(x) + 1
# class SelfAttn(nn.Module):
#     def __init__(self, dim, num_heads=8, bias=False):
#         super(SelfAttn, self).__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=bias)
#         self.proj_out = nn.Linear(dim, dim)
#
#     def forward(self, x):
#         b, N, c = x.shape
#
#         qkv = self.qkv(x).chunk(3, dim=-1)
#         # [b, N, c] -> [b, N, head, c//head] -> [b, head, N, c//head]
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
#         Q = elu_feature_map(q)
#         K = elu_feature_map(k)
#         v_length = v.size(1)
#         v=v/v_length
#          # prevent fp16 overflow
#         KV = torch.einsum("nshd,nshv->nhdv", K, v)  # (S,D)' @ S,V
#         Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + 1e-6)
#         queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length
#         queried_values=queried_values.contiguous()
#         x = rearrange(queried_values, 'b i j c -> b j (i c)')
#         x = self.proj_out(x)
#         return x

# [b, head, N, c//head] * [b, head, N, c//head] -> [b, head, N, N]
# attn = torch.einsum('bijc, bikc -> bijk', q, k) * self.scale
# attn = attn.softmax(dim=-1)
# # [b, head, N, N] * [b, head, N, c//head] -> [b, head, N, c//head] -> [b, N, head, c//head]
# x = torch.einsum('bijk, bikc -> bijc', attn, v)
# x = rearrange(x, 'b i j c -> b j (i c)')
# x = self.proj_out(x)
# return x

class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(SelfAttn, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, N, c = x.shape

        qkv = self.qkv(x).chunk(3, dim=-1)
        # [b, N, c] -> [b, N, head, c//head] -> [b, head, N, c//head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        # [b, head, N, c//head] * [b, head, N, c//head] -> [b, head, N, N]
        attn = torch.einsum('bijc, bikc -> bijk', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        # [b, head, N, N] * [b, head, N, c//head] -> [b, head, N, c//head] -> [b, N, head, c//head]
        x = torch.einsum('bijk, bikc -> bijc', attn, v)
        x = rearrange(x, 'b i j c -> b j (i c)')
        x = self.proj_out(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x,H,W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size
    Returns:
        windows: (num_windows*b, window_size, window_size, c) [non-overlap]
    """
    return rearrange(x, 'b (h s1) (w s2) c -> (b h w) s1 s2 c', s1=window_size, s2=window_size)


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image
    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    return rearrange(windows, '(b h w) s1 s2 c -> b (h s1) (w s2) c', b=b, h=h // window_size, w=w // window_size)


class JConv(nn.Module):
    """
    Next Convolution Block
    """

    def __init__(self, in_channels, out_channels, path_dropout=0.,
                 drop=0., mlp_ratio=2,isfirst=False):
        super(JConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isfirst == False:
            self.patch_embed = CBR(in_channels, out_channels)
        else:
            self.patch_embed = CBD0(in_channels, out_channels)
        self.mhca = MHCA(out_channels)
        self.attention_path_dropout = DropPath(path_dropout)
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.mlp = MLP(out_channels, mlp_ratio=mlp_ratio, drop=drop, bias=True)
        self.mlp_path_dropout = DropPath(path_dropout)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.attention_path_dropout(self.mhca(x))
        out = self.norm(x)
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x


class JConvx(nn.Module):
    """
    Next Convolution Block
    """

    def __init__(self, in_channels, out_channels, path_dropout=0.,
                 drop=0., mlp_ratio=2,isfirst=False):
        super(JConvx, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isfirst == False:
            self.patch_embed = JCBR(in_channels, out_channels)
        else:
            self.patch_embed = CBD0(in_channels, out_channels)
        self.mhca = MHCA(out_channels)
        self.attention_path_dropout = DropPath(path_dropout)
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.mlp = MLP(out_channels, mlp_ratio=mlp_ratio, drop=drop, bias=True)
        self.mlp_path_dropout = DropPath(path_dropout)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.attention_path_dropout(self.mhca(x))
        out = self.norm(x)
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x
class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class MLP(nn.Module):
    def __init__(self, in_features, mlp_ratio=None, drop=0., bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features * mlp_ratio, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_features * mlp_ratio, in_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """

    def __init__(self, out_channels):
        super(MHCA, self).__init__()
        self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=out_channels, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(80, 60), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / d_model // 2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


class LoFTREncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(LoFTREncoderLayer, self).__init__()
        self.dim = d_model // nhead
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model , bias=False),
            nn.ReLU(True),
            nn.Linear(d_model , d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class Transformer(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, mlp_ratio=4, qkv_bias=False):
        super(Transformer, self).__init__()
        self.window_size = window_size
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttn(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = rearrange(x, 'b c h w -> b h w c')
        b, h, w, c = x.shape

        shortcut = x
        x = self.norm1(x)

        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, c
        x_windows = rearrange(x_windows, 'B s1 s2 c -> B (s1 s2) c', s1=self.window_size,
                              s2=self.window_size)  # nW*b, window_size*window_size, c

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*b, window_size*window_size, c

        # merge windows
        attn_windows = rearrange(attn_windows, 'B (s1 s2) c -> B s1 s2 c', s1=self.window_size, s2=self.window_size)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # b H' W' c

        # reverse cyclic shift
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x + shortcut
        x = x + self.mlp(self.norm2(x))
        return rearrange(x, 'b h w c -> b c h w')


class CrossMatch(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self):
        super(CrossMatch, self).__init__()
        self.d_model = 256
        self.nhead = 8
        self.pos_encoding = PositionEncodingSine(256, temp_bug_fix=False)

        self.cross_layer0 = LoFTREncoderLayer(256, 8)
        self.cross_layer1 = LoFTREncoderLayer(256, 8)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        h = feat0.shape[2]
        w = feat0.shape[3]
        feat0 = rearrange(self.pos_encoding(feat0), 'n c h w -> n (h w) c')
        feat1 = rearrange(self.pos_encoding(feat1), 'n c h w -> n (h w) c')

        feat0 = self.cross_layer0(feat0, feat1)
        feat0 = self.cross_layer1(feat0, feat1)
        feat0 = rearrange(feat0, 'n (h w) c -> n c h w', h=h, w=w)
        # feat1 = self.cross_layer1(feat1, feat0)

        return feat0


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)
        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class Cross(nn.Module):
    def __init__(self, dim, mlp_ratio=2, qkv_bias=False):
        super(Cross, self).__init__()
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)
        # self.m=nn.Linear(1200, 1200)

    def forward(self, featvi, attention):
        # attention=self.m(attention)
        shortcut = featvi
        featvi = self.norm1(featvi)
        featvi = self.qkv(featvi)

        x = torch.einsum('nl, nlc -> nlc', attention, featvi)
        x = self.proj_out(x)  # nW*b, window_size*window_size, c
        x = x + shortcut
        x = x + self.mlp(self.norm2(x))
        return x
#
# class Shape(nn.Module):
#     def __init__(self):
#         super(Shape, self).__init__()
#         self.encoder1 = NCB(2, 64)
#         self.encoder2=NCB(64,256)
#         self.trans = nn.Sequential(nn.Linear(1200, 2400), nn.Linear(2400, 1200))
#         # self.trans2 = nn.Sequential(nn.Linear(1200, 2400), nn.Linear(2400, 1200))
#         # self.m=nn.Linear(1200, 1200)
#
#     def forward(self, maskir, maskvi):
#         # attention=self.m(attention)
#         xs = torch.linspace(0, 30 - 1, 30)
#         ys = torch.linspace(0, 40 - 1, 40)
#         xs = xs / (30 - 1)
#         ys = ys / (40 - 1)
#         grid = torch.stack(torch.meshgrid([xs, ys]), dim=-1).unsqueeze(0).repeat(int(maskir.shape[0]), 1, 1, 1).cuda()
#
#         h=grid.shape[1]
#         w=grid.shape[2]
#         grid = rearrange(grid, 'n h w c -> n c h w')
#         grid = self.encoder1(grid)
#         # grid = rearrange(grid, 'n c h w -> n (h w) c')
#         # maskir = rearrange(maskir, 'n c h w -> n (h w) c')
#         # maskvi = rearrange(maskir, 'n c h w -> n (h w) c')
#
#         gridir = grid * maskir
#         gridvi = grid * maskvi
#         gridir=self.encoder2(gridir)
#         gridvi = self.encoder2(gridvi)
#         gridir, gridvi = rearrange(gridir, 'n c h w -> n c (h w)'), rearrange(gridvi, 'n c h w -> n c (h w)')
#         gridir = self.trans(gridir)+gridir
#         gridvi = self.trans(gridvi)+gridvi
#         gridir, gridvi = rearrange(gridir, 'n c (h w) -> n c h w',h=h,w=w), rearrange(gridvi, 'n c (h w) -> n c h w',h=h,w=w)
#         return gridir, gridvi


class Shape(nn.Module):
    def __init__(self):
        super(Shape, self).__init__()
        self.encoder1 = JConv(2, 256)
        self.encoder2=JConv(256,256)
        self.trans1 = LoFTREncoderLayer(256, 8)
        # xs = torch.linspace(0, 30 - 1, 30)
        # ys = torch.linspace(0, 40 - 1, 40)
        # xs = xs / (30 - 1)
        # ys = ys / (40 - 1)
        # xs = torch.linspace(0, 30 - 1, 30)
        # ys = torch.linspace(0, 40 - 1, 40)
        # xs = xs / (30 - 1)
        # ys = ys / (40 - 1)
        # xs = torch.linspace(0, 45 - 1, 45)
        # ys = torch.linspace(0, 60 - 1, 60)
        # xs = xs / (45 - 1)
        # ys = ys / (60 - 1)
        xs = torch.linspace(0, 60 - 1, 60)
        ys = torch.linspace(0, 80 - 1, 80)
        xs = xs / (60 - 1)
        ys = ys / (80 - 1)
        self.grid = torch.stack(torch.meshgrid([xs, ys]), dim=-1).unsqueeze(0).repeat(1, 1, 1, 1).cuda()

        # self.trans2 = LoFTREncoderLayer(256, 8)
        # self.trans3 = LoFTREncoderLayer(256, 8)
        # self.encoder3 = NCB(64, 256)
        # self.trans2 = nn.Sequential(nn.Linear(1200, 2400), nn.Linear(2400, 1200))
        # self.m=nn.Linear(1200, 1200)


    def forward(self, maskir, maskvi):
        # attention=self.m(attention)
        # xs = torch.linspace(0, 30 - 1, 30)
        # ys = torch.linspace(0, 40 - 1, 40)
        # xs = xs / (30 - 1)
        # ys = ys / (40 - 1)
        # grid = torch.stack(torch.meshgrid([xs, ys]), dim=-1).unsqueeze(0).repeat(int(maskir.shape[0]), 1, 1, 1).cuda()

        h=self.grid.shape[1]
        w=self.grid.shape[2]
        grid = rearrange(self.grid, 'n h w c -> n c h w')
        grid = self.encoder1(grid)
        # grid = rearrange(grid, 'n c h w -> n (h w) c')
        # maskir = rearrange(maskir, 'n c h w -> n (h w) c')
        # maskvi = rearrange(maskir, 'n c h w -> n (h w) c')

        gridir = grid * maskir
        gridvi = grid * maskvi
        gridir=self.encoder2(gridir)
        gridvi = self.encoder2(gridvi)
        gridir, gridvi = rearrange(gridir, 'n c h w -> n (h w) c'), rearrange(gridvi, 'n c h w -> n (h w) c')
        gridir = self.trans1(gridir,gridir)
        gridvi = self.trans1(gridvi,gridvi)
        # gridir = self.trans2(gridir, gridvi)
        # gridvi = self.trans3(gridvi, gridir)
        gridir, gridvi = rearrange(gridir, 'n (h w) c -> n c h w',h=h,w=w), rearrange(gridvi, 'n (h w) c -> n c h w',h=h,w=w)
        # gridir = self.encoder3(gridir)
        # gridvi = self.encoder3(gridvi)
        return gridir, gridvi
# class Shape(nn.Module):
#     def __init__(self):
#         super(Shape, self).__init__()
#         self.encoder = nn.Sequential(NCB(2, 64), NCB(64, 256))
#         self.trans = nn.Sequential(nn.Linear(1200, 2400), nn.Linear(2400, 1200))
#         # self.m=nn.Linear(1200, 1200)
#
#     def forward(self, maskir, maskvi):
#         # attention=self.m(attention)
#         xs = torch.linspace(0, 30 - 1, 30)
#         ys = torch.linspace(0, 40 - 1, 40)
#         xs = xs / (30 - 1)
#         ys = ys / (40 - 1)
#         grid = torch.stack(torch.meshgrid([xs, ys]), dim=-1).unsqueeze(0).repeat(int(maskir.shape[0]), 1, 1, 1).cuda()
#
#         h=grid.shape[1]
#         w=grid.shape[2]
#         grid = rearrange(grid, 'n h w c -> n c h w')
#         grid = self.encoder(grid)
#         # grid = rearrange(grid, 'n c h w -> n (h w) c')
#         # maskir = rearrange(maskir, 'n c h w -> n (h w) c')
#         # maskvi = rearrange(maskir, 'n c h w -> n (h w) c')
#
#         gridir = grid *maskir
#         gridvi = grid *maskvi
#         gridir, gridvi = rearrange(gridir, 'n c h w -> n c (h w)'), rearrange(gridvi, 'n c h w -> n c (h w)')
#         gridir = self.trans(gridir)
#         gridvi = self.trans(gridvi)
#         gridir, gridvi = rearrange(gridir, 'n c (h w) -> n c h w',h=h,w=w), rearrange(gridvi, 'n c (h w) -> n c h w',h=h,w=w)
#         return gridir, gridvi


if __name__ == '__main__':
    device = torch.device('cuda')
    model = ResNetFPN_8_2().to(device)
    a = torch.rand((3, 1, 240, 320), dtype=torch.float32).cuda()
    result = model(a)
    from mmcv.cnn import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (1, 320, 240))
    print(flops)
    print(params)


