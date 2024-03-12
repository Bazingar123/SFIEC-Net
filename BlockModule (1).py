import time

import torch
import math
import torch.nn as nn
from basicModule import *
import numpy as np

class SSB(nn.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale, n_subs, conv=default_conv):
        super(SSB, self).__init__()
        self.spa = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale, n_subs=n_subs)
        self.spc = ResAttentionBlock(conv, n_feats, 1, act=act, res_scale=res_scale, n_subs=n_subs)

    def forward(self, x):
        return self.spc(self.spa(x))

class SB1(nn.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale, n_subs, dilation, fromLast=True, conv=default_conv):
        super(SB1, self).__init__()
        self.spa1 = ResBlock_st1(conv, n_feats, kernel_size, act=act, n_subs=n_subs, fromLast=fromLast, dilation=dilation)
        self.spa2 = ResBlock_st1(conv, n_feats, kernel_size, act=act, n_subs=n_subs, fromLast=fromLast, dilation=dilation)

    def forward(self, x):
        return x + self.spa2(self.spa1(x))

class SB1_easpp(nn.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale, n_subs, dilation=1, fromLast=True, conv=default_conv):
        super(SB1_easpp, self).__init__()
        # self.down = nn.Conv2d(n_feats, n_feats // 4, 1, padding=0, bias=True, groups=n_subs)
        self.spa1 = ResBlock_st1(conv, n_feats, kernel_size, act=act, n_subs=n_subs, fromLast=fromLast, dilation=1)
        self.spa2 = ResBlock_st1(conv, n_feats, kernel_size, act=act, n_subs=n_subs, fromLast=fromLast, dilation=4)
        self.spa3 = ResBlock_st1(conv, n_feats, kernel_size, act=act, n_subs=n_subs, fromLast=fromLast, dilation=8)
        self.convFuse = GroupChannelFuse(n_feats, n_subs, 3)
        # self.up = nn.Conv2d(n_feats // 4, n_feats, 1, padding=0, bias=True, groups=n_subs)

    def forward(self, x):
        # y = self.down(x)
        y1 = self.spa1(x)
        y2 = self.spa2(x)
        y3 = self.spa3(x)
        yt = self.convFuse(y1, y2, y3)
        return x + yt

class SB1_fuse(nn.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale, n_subs, dilation, fromLast=True, conv=default_conv):
        super(SB1_fuse, self).__init__()
        self.spa1 = ResBlock_st1(conv, n_feats, kernel_size, act=act, n_subs=n_subs, fromLast=fromLast, dilation=dilation)
        # self.spa2 = ResBlock_st1(conv, n_feats, kernel_size, act=act, n_subs=n_subs, fromLast=fromLast)

    def forward(self, x):
        return self.spa1(x)

class SB2(nn.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale, n_subs, dilation, fromLast=True, conv=default_conv):
        super(SB2, self).__init__()
        self.spa1 = ResBlock_st2(conv, n_feats, n_feats, kernel_size, act=act, n_subs=n_subs, fromLast=fromLast, dilation=dilation)
        self.spa2 = ResBlock_st2(conv, n_feats, n_feats, kernel_size, act=act, n_subs=n_subs, fromLast=fromLast, dilation=dilation)

    def forward(self, x):
        return x + self.spa2(self.spa1(x))


class SSPN(nn.Module):
    def __init__(self, n_feats, n_blocks, act, res_scale, n_subs):
        super(SSPN, self).__init__()
        kernel_size = 3

        self.net1 = SSB(n_feats, kernel_size, act=act, res_scale=res_scale, n_subs=n_subs)
        self.net2 = SSB(n_feats, kernel_size, act=act, res_scale=res_scale, n_subs=n_subs)
        self.net3 = SSB(n_feats, kernel_size, act=act, res_scale=res_scale, n_subs=n_subs)

    def forward(self, x):
        res = self.net1(x)
        # print("SSPN.shape", res.shape)
        # save_dir = "SSPN_" + "_SSB1_" + str(res.shape[2]) + "_" + time.ctime() + ".npy"
        # np.save(save_dir, res.cpu().detach().numpy())
        res = self.net2(res)
        # save_dir = "SSPN_" + "_SSB2_" + str(res.shape[2]) + "_" + time.ctime() + ".npy"
        # np.save(save_dir, res.cpu().detach().numpy())
        res = self.net3(res)
        # save_dir = "SSPN_" + "_SSB3_" + str(res.shape[2]) + "_" + time.ctime() + ".npy"
        # np.save(save_dir, res.cpu().detach().numpy())
        res += x
        return res


class SPN(nn.Module):
    def __init__(self, n_feats, n_blocks, act, res_scale, n_subs):
        super(SPN, self).__init__()
        kernel_size = 3

        self.net1 = SB1_easpp(n_feats, kernel_size, act=act, res_scale=res_scale, n_subs=n_subs, fromLast=False)
        self.net2 = SB1(n_feats, kernel_size, act=act, res_scale=res_scale, n_subs=n_subs, dilation=1)
        self.net3 = SB1(n_feats, kernel_size, act=act, res_scale=res_scale, n_subs=n_subs, dilation=1)
        # self.net4 = SB1(n_feats, kernel_size, act=act, res_scale=res_scale, n_subs=n_subs, dilation=1)
        # self.net3 = SB1(n_feats, kernel_size, act=act, res_scale=res_scale, n_subs=n_subs, dilation=12)
        # self.net4 = SB1_fuse(n_feats, 1, act=act, res_scale=res_scale, n_subs=n_subs, dilation=1)
        # self.groupFuse = GroupChannelFuse(n_feats, n_subs, 3)

    def forward(self, x):
        res1 = self.net1(x)
        res2 = self.net2(res1)
        res3 = self.net3(res2)
        # res4 = self.net4(res3)
        # res = self.groupFuse(res1, res2, res3)
        # res = self.net4(res)
        return res3 + x

class SPN2(nn.Module):
    def __init__(self, n_feats, n_blocks, act, res_scale, n_subs):
        super(SPN2, self).__init__()
        kernel_size = 3

        self.net1 = SB2(n_feats, kernel_size, act=act, res_scale=res_scale, n_subs=n_subs, fromLast=False, dilation=1)
        self.net2 = SB2(n_feats, kernel_size, act=act, res_scale=res_scale, n_subs=n_subs, dilation=1)
        self.net3 = SB2(n_feats, kernel_size, act=act, res_scale=res_scale, n_subs=n_subs, dilation=1)
        # self.net4 = SB2(n_feats, kernel_size, act=act, res_scale=res_scale, n_subs=n_subs, dilation=1)
        # self.groupFuse = GroupChannelFuse(n_feats, n_subs, 3) 

    def forward(self, x):
        res = self.net1(x)
        res = self.net2(res)
        # res = self.net3(res)
        # res = self.groupFuse(res1, res2, res3)
        # res4 = self.net4(res3)
        return res + x

# a single branch of proposed SSPSR
class BranchUnit1(nn.Module):
    def __init__(self, n_colors, n_feats, n_outputs, n_blocks, act, res_scale, up_scale, n_subs, use_tail=True, conv=default_conv):
        super(BranchUnit1, self).__init__()
        kernel_size = 3
        self.head = GroupFE(n_colors, n_feats, kernel_size, bias=True, n_subs=n_subs)
        # self.head = ResBlock_st2(conv, n_colors, n_feats, kernel_size, act=act, n_subs=n_subs, fromLast=False, dilation=1)
        self.body1 = SPN(n_feats, n_blocks, act, res_scale, n_subs=n_subs)
        # self.body2 = SPN(n_feats, n_blocks, act, res_scale, n_subs=n_subs)
        # self.body3 = SPN(n_feats, n_blocks, act, res_scale, n_subs=n_subs)
        # self.body4 = SPN(n_feats, n_blocks, act, res_scale, n_subs=n_subs)
        self.upsample = Upsampler(conv, up_scale, n_feats, n_subs)
        self.tail = None
        self.act = act
        self.tail2 = None
        self.CA2 = DGSCALayer(n_feats, n_subs)
        if use_tail:
            if n_feats != n_outputs:
                self.tail = conv(n_feats, n_feats, kernel_size, n_subs)
                self.tail2 = conv(n_feats, n_outputs, 1, n_subs)
            else:
                self.tail = conv(n_feats, n_outputs, kernel_size, n_subs)

    def forward(self, x):
        y = self.head(x)
        # print("Bran1.shape", y.shape)
        # save_dir = "bran1_" + "_head_" + str(y.shape[2]) + "_" + time.ctime() + ".npy"
        # np.save(save_dir, y.cpu().detach().numpy())
        y = self.body1(y)
        # y = self.upsample(y)
        # print("head.shape", y.shape)
        # save_dir = "branch1_" + "_Up_" + str(y.shape[2]) + "_" + time.ctime() + ".npy"
        # np.save(save_dir, y.cpu().detach().numpy())
        # y = self.CA2(y)
        if self.tail is not None:
            y = self.tail(y)
            if self.tail2 is not None:
                y = self.tail2(y)


        return y
    
class BranchUnit2(nn.Module):
    def __init__(self, n_colors, n_feats, n_outputs, n_blocks, act, res_scale, n_subs, use_tail=True, conv=default_conv):
        super(BranchUnit2, self).__init__()
        kernel_size = 3
        self.head = nn.Conv2d(n_colors, n_feats, kernel_size, padding=(kernel_size//2), bias=True, groups=n_subs)
        self.body = SSPN(n_feats, n_blocks, act, res_scale, n_subs=n_subs)
        self.tail = None
        if use_tail:
            self.tail = conv(n_feats, n_outputs, 1, n_subs)    #9999

    def forward(self, x):
        # y = self.head(x)
        y = self.body(x)
        if self.tail is not None:
            y = self.tail(y)

        return y

class BranchUnit3(nn.Module):
    def __init__(self, n_colors, n_feats, n_outputs, n_blocks, act, res_scale, up_scale, n_subs, use_tail=True, conv=default_conv):
        super(BranchUnit3, self).__init__()
        kernel_size = 3
        self.head = nn.Conv2d(n_colors, n_feats, kernel_size, padding=(kernel_size//2), bias=True, groups=n_subs)
        # self.head = nn.Conv2d(n_colors, n_feats, kernel_size, padding=(kernel_size//2), bias=True)
        self.body1 = SPN(n_feats, n_blocks, act, res_scale, n_subs=n_subs)
        # self.body2 = SPN(n_feats, n_blocks, act, res_scale, n_subs=n_subs)
        # self.body3 = SPN2(n_feats, n_blocks, act, res_scale, n_subs=n_subs)
        self.upsample = Upsampler(conv, up_scale, n_feats, n_subs)
        self.tail = None
        self.tail2 = None
        self.act = act
        self.CA = CALayer(n_feats)
        self.CA2 = DGSCALayer(n_feats, n_subs)
        if use_tail:
            if n_feats != n_outputs:
                self.tail = conv(n_feats, n_feats, kernel_size, n_subs)
                self.tail2 = conv(n_feats, n_outputs, 1, n_subs)
            else:
                self.tail = conv(n_feats, n_outputs, kernel_size, n_subs)

    def forward(self, x):
        # y = self.head(x)
        # print("Bran2.shape", y.shape)
        # save_dir = "bran2" + "_head_" + str(y.shape[2]) + "_" + time.ctime() + ".npy"
        # np.save(save_dir, y.cpu().detach().numpy())
        y = self.body1(x)
        # save_dir = "bran2" + "_body_" + str(y.shape[2]) + "_" + time.ctime() + ".npy"
        # np.save(save_dir, y.cpu().detach().numpy())
        y = self.upsample(y)
        # save_dir = "bran2" + "_up_" + str(y.shape[2]) + "_" + time.ctime() + ".npy"
        # np.save(save_dir, y.cpu().detach().numpy())
        # y = self.CA2(y)
        # save_dir = "bran2" + "_DCA_" + str(y.shape[2]) + "_" + time.ctime() + ".npy"
        # np.save(save_dir, y.cpu().detach().numpy())
        if self.tail is not None:
            y = self.tail(y)
            if self.tail2 is not None:
                y = self.tail2(y)
            # save_dir = "bran2" + "_tail_" + str(y.shape[2]) + "_" + time.ctime() + ".npy"
            # np.save(save_dir, y.cpu().detach().numpy())

        return y

class Swish(nn.Module):
    def __int__(self, inplace=True):
        super(Swish, self).__int__()
        self.inplace = inplace

    def forward(self, x):
        x.mul_(torch.sigmoid(x))
        return x

# class DeepShare(nn.Module):
#     def __init__(self, n_subs, n_ovls, n_colors, n_blocks, n_feats, n_scale, res_scale, use_share=True, conv=default_conv) -> object:
#         super(DeepShare, self).__init__()
#         kernel_size = 3
#
#         self.shared = use_share
#         self.act = nn.ReLU(inplace=True)
#
#         if self.shared:
#             self.branch1 = BranchUnit1(n_colors, n_feats, n_feats, n_blocks, self.act, res_scale, up_scale=n_scale//2, conv=default_conv, use_tail=False, n_subs=n_colors)
#             self.branch2 = BranchUnit1(n_feats, n_feats, n_colors, n_blocks, self.act, res_scale, up_scale=2, conv=default_conv, n_subs=n_colors)
#
#             # up_scale=n_scale//2 means that we upsample the LR input n_scale//2 at the branch network, and then conduct 2 times upsampleing at the global network
#         else:
#             print("Not impletments!!!")
#
#         self.trunk = BranchUnit2(n_colors, n_feats, n_feats, n_blocks, self.act, res_scale, use_tail=True, conv=default_conv, n_subs=n_colors)
#
#         self.skip_conv = conv(n_colors, n_feats, kernel_size, 9999)
#         # self.skip_conv = conv(n_colors, n_feats, kernel_size)
#
#         self.final = conv(n_feats, n_colors, kernel_size, 9999)
#         # self.final = conv(n_feats, n_colors, kernel_size)
#         self.sca = n_scale
#
#     def forward(self, x, lms, modality):
#         b, c, h, w = x.shape
#
#         # the rest steps depend on the modality which could be spectral images or RGB images
#         if modality == "spectral":
#             # Initialize intermediate “result”, which is upsampled with n_scale times
#             y = torch.zeros(b, c, self.sca * h, self.sca * w).cuda()
#
#             if self.shared:
#                 x = self.branch1(x)
#                 y1 = self.branch2(x)
#             else:
#                 print("Wrong! Not implemented error")
#
# #                 y[:, sta_ind:end_ind, :, :] += xi
# #                 channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1
#
# #             # intermediate “result” is averaged according to their spectral indices
# #             y = y / channel_counter.unsqueeze(1).unsqueeze(2)
#
#             y = self.trunk(y1)
#             y = y + self.skip_conv(lms)
#             y = self.final(y)
#
#         elif modality == "rgb":
#
#             y = self.branch1(x)
#             y = self.branch2(y)
#             y = self.trunk_RGB(y)
#
#             y = y + self.skip_conv_RGB(lms)
#             y = self.final_RGB(y)
#
#         else:
#             raise("Not implemented!!!")
#         return y


class DeepShare2(nn.Module):
    def __init__(self, n_subs, n_ovls, n_colors, n_blocks, n_feats, n_scale, res_scale, use_share=True, conv=default_conv) -> object:
        super(DeepShare2, self).__init__()
        kernel_size = 3

        self.shared = use_share
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.LeakyReLU(n_feats)
        self.lms = conv(n_colors, n_feats // 4, 1, n_colors)
        self.sca = n_scale
        self.final = conv(n_feats // 4, n_colors, 1, n_colors)  #11.27 noon kernel_size=1 n_subs=n_colors

        if self.shared:
            self.branch1 = BranchUnit1(n_colors, n_feats, n_feats // 4, n_blocks, self.act, res_scale, up_scale=n_scale//2, conv=default_conv, n_subs=n_colors)  #11.27 noon use_share=True
            # self.branch2 = BranchUnit3(n_feats, n_feats, n_feats, n_blocks, self.act, res_scale, up_scale=2, conv=default_conv, n_subs=n_colors)
            self.branch2 = BranchUnit3(n_feats // 4, n_feats // 4, n_feats // 4, n_blocks, self.act, res_scale, up_scale=n_scale, conv=default_conv, n_subs=n_colors)
            # self.branch2 = BranchUnit2(n_feats, n_feats, n_feats, n_blocks, self.act, res_scale, up_scale=2, conv=default_conv, n_subs=n_colors)
            self.branch3 = BranchUnit2(n_feats // 4, n_feats // 4, n_feats // 4, n_blocks, self.act, res_scale, conv=default_conv, n_subs=n_colors)

    def forward(self, x, lms, modality):

        if self.shared:
            x = self.branch1(x)
            y = self.branch2(x)
            y = self.branch3(y)
            y = y + self.lms(lms)
        else:
            print("Wrong! Not implemented error")

        y = self.final(y)

        return y

class DeepShare3(nn.Module):
    def __init__(self, n_subs, n_ovls, n_colors, n_blocks, n_feats, n_scale, res_scale, use_share=True, conv=default_conv) -> object:
        super(DeepShare3, self).__init__()
        kernel_size = 3
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        self.start_idx = []
        self.end_idx = []
        self.sub_fe_size = n_feats // n_colors
        self.sub_size = self.sub_fe_size * n_subs
        self.final = conv(n_feats, n_colors, 1, n_colors)

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            self.start_idx.append(sta_ind * self.sub_fe_size)
            self.end_idx.append(end_ind * self.sub_fe_size)

        self.shared = use_share
        self.act = nn.ReLU(inplace=True)
        self.sca = n_scale

        if self.shared:
            self.branch1 = BranchUnit1(n_subs, self.sub_size, self.sub_size, n_blocks, self.act, res_scale, up_scale=n_scale//2, conv=default_conv, n_subs=n_subs)
            self.branch2 = BranchUnit3(self.sub_size, self.sub_size, self.sub_size, n_blocks, self.act, res_scale, up_scale=2, conv=default_conv, n_subs=n_subs)

    def forward(self, x, lms, modality):
        b, c, h, w = x.shape
        c = c * self.sub_fe_size
        # the rest steps depend on the modality which could be spectral images or RGB images
        # Initialize intermediate “result”, which is upsampled with n_scale times
        y = torch.zeros(b, c, self.sca * h, self.sca * w).cuda()
        channel_counter = torch.zeros(c).cuda()

        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]
            xi = x[:, sta_ind // self.sub_fe_size:end_ind // self.sub_fe_size, :, :]

            if self.shared:
                xi = self.branch1(xi)
                xi = self.branch2(xi)
            else:
                print("Wrong! Not implemented error")

            y[:, sta_ind:end_ind, :, :] += xi
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1

        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        y = self.final(y)
        return y