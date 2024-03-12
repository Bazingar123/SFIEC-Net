import torch.nn as nn
import math
import torch

# def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
#     if dilation==1:
#        return nn.Conv2d(
#            in_channels, out_channels, kernel_size,
#            padding=(kernel_size//2), bias=bias)
#     elif dilation==2:
#        return nn.Conv2d(
#            in_channels, out_channels, kernel_size,
#            padding=2, bias=bias, dilation=dilation)

#     else:
#        return nn.Conv2d(
#            in_channels, out_channels, kernel_size,
#            padding=3, bias=bias, dilation=dilation)

def default_conv(in_channels, out_channels, kernel_size, n_subs, bias=True, dilation=1):
    if(n_subs==9999):
        return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias)
    if dilation==1:
        return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias, groups=n_subs)
    elif dilation==2:
        return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=2, bias=bias, dilation=dilation, groups=n_subs)

    else:
        return nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=dilation, bias=bias, dilation=dilation, groups=n_subs)

class HDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_subs, bias=True,):
        super(HDC, self).__init__()
        self.m = []
        self.m.append(nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias, groups=n_subs).to("cuda"))
        self.m.append(nn.Conv2d(
           out_channels, out_channels, kernel_size,
           padding=2, bias=bias, dilation=2, groups=n_subs).to("cuda"))
        self.m.append(nn.Conv2d(
           out_channels, out_channels, kernel_size,
           padding=3, bias=bias, dilation=3, groups=n_subs).to("cuda"))
        self.body = nn.Sequential(*self.m)

    def forward(self, x):
        y = self.body(x)
        return y

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_subs, bias=True):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=(kernel_size//2), bias=bias, groups=n_subs).to("cuda")
        self.conv2 = nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=4, bias=bias, dilation=4, groups=n_subs).to("cuda")
        self.conv3 = nn.Conv2d(
           in_channels, out_channels, kernel_size,
           padding=8, bias=bias, dilation=8, groups=n_subs).to("cuda")
        self.convFuse = GroupChannelFuse(out_channels, n_subs, 3)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)
        y = self.convFuse(y1, y2, y3)
        return y

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class GroupChannelFuse(nn.Module):

    def __init__(self, n_feats, n_subs, groupNum, bias=True):
        super(GroupChannelFuse, self).__init__()

        self.n_subs = n_subs
        self.n_groupLen = n_feats//n_subs
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.n_groupLen * groupNum,
                              out_channels=self.n_groupLen,
                              kernel_size=1,
                              bias=self.bias)
        self.bn = nn.BatchNorm2d(n_feats)


    def forward(self, x, y, r=None, z=None):
        b, c, h, w = x.size()
        output_inner = []
        for i in range(self.n_subs):
            end = i * self.n_groupLen + self.n_groupLen
            if(end >= c):
                if(r is not None):
                    if(z is not None):
                        inputx = torch.cat([x[:, i*self.n_groupLen:, :, :], y[:, i*self.n_groupLen:, :, :],
                                            r[:, i*self.n_groupLen:, :, :], z[:, i*self.n_groupLen:, :, :]], dim=1)
                    else:
                        inputx = torch.cat([x[:, i*self.n_groupLen:, :, :], y[:, i*self.n_groupLen:, :, :],
                                            r[:, i*self.n_groupLen:, :, :]], dim=1)
                else:
                    inputx = torch.cat([x[:, i*self.n_groupLen:, :, :], y[:, i*self.n_groupLen:, :, :]], dim=1)
            else:
                if(r is not None):
                    if(z is not None):
                        inputx = torch.cat([x[:, i*self.n_groupLen:end, :, :], y[:, i*self.n_groupLen:end, :, :],
                                            r[:, i*self.n_groupLen:end, :, :], z[:, i*self.n_groupLen:end, :, :]], dim=1)
                    else:
                        inputx = torch.cat([x[:, i*self.n_groupLen:end, :, :], y[:, i*self.n_groupLen:end, :, :],
                                            r[:, i*self.n_groupLen:end, :, :]], dim=1)
                else:
                    inputx = torch.cat([x[:, i*self.n_groupLen:end, :, :], y[:, i*self.n_groupLen:end, :, :]], dim=1)
            res = self.conv(inputx)
            # combined_conv = self.bn2(combined_conv)
            output_inner.append(res)
        layer_output = torch.cat(output_inner, dim=1)
        layer_output = self.bn(layer_output)
        return layer_output



class DGSCALayer(nn.Module):
    def __init__(self, channel, n_subs):
        super(DGSCALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.groupLen = channel // n_subs
        self.n_subs = n_subs
        self.conv_du_GIn = nn.Sequential(
                self.avg_pool,
                nn.Conv2d(channel, n_subs, 1, padding=0, bias=True, groups=n_subs),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_subs, channel, 1, padding=0, bias=True, groups=n_subs),
                nn.Sigmoid()
        )
        self.conv_du_GOut = nn.Sequential(
                nn.Conv2d(channel, n_subs, 3, padding=0, bias=True, groups=n_subs),
                nn.ReLU(inplace=True),
                self.avg_pool,
                nn.Conv2d(n_subs, 1, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, n_subs, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du_GIn(x)
        r = self.conv_du_GOut(x)
        res = []
        for i in range(self.n_subs):
            w = y[:, i * self.groupLen : i * self.groupLen + self.groupLen, :, :] * r[:, i, :, :].unsqueeze(1)
            res.append(w)
        res = torch.cat(res, dim=1)
        return x * res

class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, n_subs, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias, n_subs=n_subs))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class ResBlock_st2(nn.Module):
    def __init__(self, conv, in_n_feats, out_n_feats, kernel_size, n_subs, dilation, fromLast=True, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock_st2, self).__init__()
        self.n_subs = n_subs
        self.in_groupLen = in_n_feats // n_subs
        self.out_groupLen = out_n_feats // n_subs
        self.m = []
        self.m2 = []
        self.act = act
        for _ in range(n_subs):
            self.m.append(
                ConvLSTMCell_st2(self.in_groupLen, self.out_groupLen, kernel_size, bias=bias, dilation=dilation))
            self.m2.append(
                ConvLSTMCell_st2(self.in_groupLen, self.out_groupLen, kernel_size, bias=bias, dilation=dilation))
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_n_feats)
        self.ChannelConv = GroupChannelFuse(out_n_feats, n_subs, 2)
        self.res_scale = res_scale
        # self.conv = nn.Conv2d(in_channels=n_feats,
        #                       out_channels=4 * n_feats,
        #                       kernel_size=kernel_size,
        #                       padding=kernel_size // 2,
        #                       bias=bias)

    def forward(self, x):
        batch, channel, height, width = x.size()
        f_inner_output = []
        b_inner_output = []
        h_f, c_f = self.init_hidden(batch_size=batch, image_size=(height, width))
        h_b, c_b = h_f, c_f
        for i in range(self.n_subs):
            back = self.n_subs - 1 - i
            f_end = i * self.in_groupLen + self.in_groupLen
            b_end = back * self.in_groupLen + self.in_groupLen
            if f_end >= channel:
                x_f = x[:, i*self.in_groupLen:, :, :]
            else:
                x_f = x[:, i*self.in_groupLen:f_end, :, :]
            if b_end >= channel:
                x_b = x[:, back*self.in_groupLen:, :, :]
            else:
                x_b = x[:, back*self.in_groupLen:b_end, :, :]
            res_h_f, res_c_f = self.m[i](x_f, h_f, c_f)
            res_h_b, res_c_b = self.m2[i](x_b, h_b, c_b)
            h_f, c_f, h_b, c_b = res_h_f, res_c_f, res_h_b, res_c_b
            f_inner_output.append(res_h_f)
            b_inner_output.append(res_h_b)
        back = []
        for i in reversed(b_inner_output):
            back.append(i)
        back_output = torch.cat(back, dim=1)
        forward_output = torch.cat(f_inner_output, dim=1)
        if self.bn is not None:
            back_output = self.bn(back_output)
            forward_output = self.bn(forward_output)
        res = self.ChannelConv(forward_output, back_output)
        # res = forward_output + back_output
        # res = res + x
        return res

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.out_groupLen, height, width).to("cuda"),
                torch.zeros(batch_size, self.out_groupLen, height, width).to("cuda"))

class ResBlock_st2_2(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, n_subs, fromLast=True, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock_st2_2, self).__init__()
        self.n_subs = n_subs
        self.groupLen = n_feats // n_subs
        self.m = []
        self.m2 = []
        self.act = act
        for _ in range(2):
            self.m.append(
                ConvLSTMCell_st2(self.groupLen, self.groupLen, kernel_size, bias=bias))
            self.m2.append(
                ConvLSTMCell_st2(self.groupLen, self.groupLen, kernel_size, bias=bias))
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(n_feats)
        self.ChannelConv = GroupChannelFuse(n_feats, n_subs, 2)
        self.res_scale = res_scale
        self.conv = nn.Conv2d(in_channels=n_feats,
                              out_channels=4 * n_feats,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              bias=bias)

    def forward(self, x):
        batch, channel, height, width = x.size()
        f_inner_output = []
        b_inner_output = []
        h_f, c_f = self.init_hidden(batch_size=batch, image_size=(height, width))
        h_b, c_b = h_f, c_f
        for i in range(self.n_subs):
            back = self.n_subs - 1 - i
            f_end = i * self.groupLen + self.groupLen
            b_end = back * self.groupLen + self.groupLen
            if f_end >= channel:
                x_f = x[:, i*self.groupLen:, :, :]
            else:
                x_f = x[:, i*self.groupLen:f_end, :, :]
            if b_end >= channel:
                x_b = x[:, back*self.groupLen:, :, :]
            else:
                x_b = x[:, back*self.groupLen:b_end, :, :]
            if(i < self.n_subs // 2):
                res_h_f, res_c_f = self.m[0](x_f, h_f, c_f)
                res_h_b, res_c_b = self.m2[0](x_b, h_b, c_b)
            elif(i > self.n_subs // 2):
                res_h_f, res_c_f = self.m[1](x_f, h_f, c_f)
                res_h_b, res_c_b = self.m2[1](x_b, h_b, c_b)
            else:
                res_h_f1, res_c_f1 = self.m[0](x_f, h_f, c_f)
                res_h_b1, res_c_b1 = self.m2[0](x_b, h_b, c_b)
                res_h_f2, res_c_f2 = self.m[1](x_f, h_f, c_f)
                res_h_b2, res_c_b2 = self.m2[1](x_b, h_b, c_b)
                res_h_f, res_c_f = (res_h_f1 + res_h_f2) / 2, (res_c_f1 + res_c_f2) / 2
                res_h_b, res_c_b = (res_h_b1 + res_h_b2) / 2, (res_c_b1 + res_c_b2) / 2
            h_f, c_f, h_b, c_b = res_h_f, res_c_f, res_h_b, res_c_b
            f_inner_output.append(res_h_f)
            b_inner_output.append(res_h_b)
        back = []
        for i in reversed(b_inner_output):
            back.append(i)
        back_output = torch.cat(back, dim=1)
        forward_output = torch.cat(f_inner_output, dim=1)
        if self.bn is not None:
            back_output = self.bn(back_output)
            forward_output = self.bn(forward_output)
        back_output, forward_output = self.act(back_output), self.act(forward_output)
        res = self.ChannelConv(forward_output, back_output)
        res = res + x
        return res

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.groupLen, height, width).to("cuda"),
                torch.zeros(batch_size, self.groupLen, height, width).to("cuda"))

class ResBlock_st1(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, n_subs, dilation, fromLast=True, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock_st1, self).__init__()
        m = []
        # for i in range(2):
        #     m.append(conv(n_feats, n_feats, kernel_size, 9999, bias=bias))
        #     if bn:
        #         m.append(nn.BatchNorm2d(n_feats))
        #     if i == 0:
        #         m.append(act)
        #顺序传递的ConvLSTM
        # m.append(ConvLSTMCell(n_feats, n_feats, kernel_size, bias=bias, n_subs=n_subs, back=False, fromLast=fromLast))
        # if bn:
        #     m.append(nn.BatchNorm2d(n_feats))
        # m.append(act)
        # m.append(ConvLSTMCell(n_feats, n_feats, kernel_size, bias=bias, n_subs=n_subs, back=True))
        # if bn:
        #     m.append(nn.BatchNorm2d(n_feats))
        # self.body = nn.Sequential(*m)
        # self.res_scale = res_scale
        #以原输入为输入的反向传播的ConvLSTM
        self.act = act
        m.append(ConvLSTMCell_st1(n_feats, n_feats, kernel_size, bias=bias, n_subs=n_subs, back=False, fromLast=fromLast, dilation=dilation))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        m2 = []
        m2.append(ConvLSTMCell_st1(n_feats, n_feats, kernel_size, bias=bias, n_subs=n_subs, back=True, fromLast=fromLast, dilation=dilation))
        if bn:
            m2.append(nn.BatchNorm2d(n_feats))
        self.ChannelConv = GroupChannelFuse(n_feats, n_subs, 2)
        self.body = nn.Sequential(*m)
        self.body2 = nn.Sequential(*m2)
        self.res_scale = res_scale
        #普通分组卷积
        # for i in range(2):
        #     m.append(conv(n_feats, n_feats, kernel_size, n_subs=n_subs, bias=bias))
        #     if bn:
        #         m.append(nn.BatchNorm2d(n_feats))
        #     if i == 0:
        #         m.append(act)
        # self.body = nn.Sequential(*m)
        # self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res2 = self.body2(x).mul(self.res_scale)
        res3 = self.ChannelConv(res, res2)
        # res3 = res + res2
        # res3 = res3 + x
        return res3


class ResBlock_st1_group(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, n_subs, fromLast=True, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock_st1_group, self).__init__()
        self.G = math.ceil((31 - 2) / (8 - 2))
        self.start_idx = []
        self.end_idx = []
        self.sub_size = (n_feats // 31) * 8
        self.bn = nn.BatchNorm2d(self.sub_size)
        self.m_f = []
        self.m_b = []
        for g in range(self.G):
            sta_ind = (8 - 2) * g
            end_ind = sta_ind + 8
            if end_ind > 31:
                end_ind = 31
                sta_ind = 31 - 8
            self.start_idx.append(sta_ind*(n_feats // 31))
            self.end_idx.append(end_ind*(n_feats // 31))
            self.m_f.append(ConvLSTMCell_st1(self.sub_size, self.sub_size, kernel_size, bias=bias, n_subs=8, back=False, fromLast=fromLast))
            self.m_b.append(ConvLSTMCell_st1(self.sub_size, self.sub_size, kernel_size, bias=bias, n_subs=8, back=True, fromLast=fromLast))
        self.ChannelConv = GroupChannelFuse(n_feats, n_subs, 2)
        self.res_scale = res_scale

    def forward(self, x):
        b, c, h, w = x.shape
        y_f = torch.zeros(b, c, h, w).cuda()
        y_b = torch.zeros(b, c, h, w).cuda()
        channel_counter = torch.zeros(c).cuda()
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]
            xi = x[:, sta_ind:end_ind, :, :]
            xi_f = self.m_f[g](xi).mul(self.res_scale)
            xi_b = self.m_b[g](xi).mul(self.res_scale)
            xi_f = self.bn(xi_f)
            xi_b = self.bn(xi_b)
            y_f[:, sta_ind:end_ind, :, :] += xi_f
            y_b[:, sta_ind:end_ind, :, :] += xi_b
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1
        y_f = y_f / channel_counter.unsqueeze(1).unsqueeze(2)
        y_b = y_b / channel_counter.unsqueeze(1).unsqueeze(2)
        res3 = self.ChannelConv(y_f, y_b)
        res3 = res3 + x
        return res3

class ResBlock_st1_GN(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, n_subs, fromLast=True, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock_st1_GN, self).__init__()
        m = []
        #普通分组卷积
        for i in range(2):
            m.append(GroupFE(n_feats, n_feats, kernel_size, n_subs=n_subs, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        return res + x

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, n_subs, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                # m.append(conv(n_feats, 4 * n_feats, 3, n_subs, bias))
                m.append(conv(n_feats, 4 * n_feats, 3, n_subs, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            # m.append(conv(n_feats, 9 * n_feats, 3, n_subs, bias))
            m.append(conv(n_feats, 9 * n_feats, 3, n_subs, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
            elif act == 'leakyrelu':
                m.append(nn.LeakyReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class ResAttentionBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, n_subs, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttentionBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias, n_subs=n_subs))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        m.append(DGSCALayer(n_feats, n_subs))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ConvLSTMCell_st2(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, dilation):

        super(ConvLSTMCell_st2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dila = dilation
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2    #使特征图大小在过程中保持一致
        self.bias = bias
        self.inception_conv = HDC(self.input_dim + self.hidden_dim, 
                                                self.hidden_dim*4,
                                                self.kernel_size,
                                                1)
        self.dila_conv = default_conv(self.input_dim + self.hidden_dim,
                                      self.hidden_dim*4,
                                      self.kernel_size,
                                      n_subs = 1,
                                      bias=self.bias,
                                      dilation=self.dila).to("cuda")
        self.conv1 = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias).to("cuda")
        self.conv = nn.Conv2d(in_channels=self.hidden_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=1,
                              # padding=self.padding,
                              bias=self.bias).to("cuda")

    def forward(self, input_tensor, hidden_state, c_state):
        input_tensor_x = self.conv1(input_tensor)
        combined = torch.cat([input_tensor_x, hidden_state], dim=1)
        combined_conv = self.conv(combined)
        # combined_conv = self.dila_conv(combined)
        # combined_conv = self.bn2(combined_conv)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)  # 在某个网站上查到，中间的参数是切完块每个块的大小
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_state + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTMCell_st1(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, n_subs, back, dilation, fromLast=True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell_st1, self).__init__()

        self.n_subs = n_subs
        self.dila = dilation
        self.fromLast = fromLast
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_groupLen = input_dim//n_subs
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2    #使特征图大小在过程中保持一致
        self.bias = bias
        self.bn = nn.BatchNorm2d(4 * self.hidden_dim)
        self.bn2 = nn.BatchNorm2d(self.n_groupLen*4)
        self.back = back
        self.inception_conv = HDC(self.n_groupLen + self.n_groupLen, 
                                                self.n_groupLen*4,
                                                self.kernel_size,
                                                1)
        
        self.dila_conv = default_conv(self.n_groupLen + self.n_groupLen,
                                      self.n_groupLen*4,
                                      kernel_size,
                                      n_subs=1,
                                      bias=bias,
                                      dilation=self.dila)
        self.aspp = ASPP(self.n_groupLen + self.n_groupLen,
                         self.n_groupLen*4,
                         self.kernel_size,
                         1)
        self.conv = nn.Conv2d(self.n_groupLen + self.n_groupLen,
                              self.n_groupLen*4,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias).to("cuda")
        self.conv1 = nn.Conv2d(self.input_dim,
                               self.input_dim,
                               kernel_size=self.kernel_size,
                               padding=self.padding,
                               bias=self.bias,
                               groups=self.n_subs).to("cuda")
        self.conv2 = nn.Conv2d(in_channels=self.n_groupLen + self.n_groupLen,
                              out_channels=self.n_groupLen*4,
                              kernel_size=self.kernel_size,
                              padding=self.dila,
                              bias=self.bias,
                              dilation=self.dila).to("cuda")

    def forward(self, input_tensor):
        b, c, h, w = input_tensor.size()
        h_cur, c_cur = self.init_hidden(batch_size=b, image_size=(h, w))

        output_inner = []
        c_inner_hidden = []
        # input_tensor = self.conv1(input_tensor)
        for i in range(self.n_subs):
            if(self.back):
                i = self.n_subs - 1 - i
            end = i * self.n_groupLen + self.n_groupLen
            if(end >= c):
                end = -1
            if end == -1:
                combined = torch.cat([input_tensor[:, i*self.n_groupLen:, :, :], h_cur], dim=1)
            else:
                combined = torch.cat([input_tensor[:, i*self.n_groupLen:end, :, :], h_cur], dim=1)
            combined_conv = self.conv2(combined)
            # combined_conv = self.aspp(combined)
            # combined_conv = self.bn2(combined_conv)
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.n_groupLen, dim=1)  # 在某个网站上查到，中间的参数是切完块每个块的大小
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)

            c_next = f * c_cur + i * g
            h_next = o * torch.tanh(c_next)
            h_cur = h_next
            c_cur = c_next
            output_inner.append(h_cur)
            c_inner_hidden.append(c_cur)

        if self.back:
            reversed_arr = []
            for i in reversed(output_inner):
                reversed_arr.append(i)
            layer_output = torch.cat(reversed_arr, dim=1)
        else:
            layer_output = torch.cat(output_inner, dim=1)
        return layer_output

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.n_groupLen, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.n_groupLen, height, width, device=self.conv.weight.device))


class GroupFE(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, n_subs, bias=True, dilation=1):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(GroupFE, self).__init__()

        self.n_subs = n_subs
        self.n_groupLen = in_channels//n_subs
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2    #使特征图大小在过程中保持一致
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.n_groupLen,
                              out_channels=out_channels // n_subs,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor):
        b, c, h, w = input_tensor.size()
        output_inner = []
        for i in range(self.n_subs):
            end = i * self.n_groupLen + self.n_groupLen
            if(end >= c):
                end = -1
            if end == -1:
                x = input_tensor[:, i*self.n_groupLen:, :, :]
            else:
                x = input_tensor[:, i*self.n_groupLen:end, :, :]
            y = self.conv(x)
            # combined_conv = self.bn2(combined_conv)
            output_inner.append(y)
        layer_output = torch.cat(output_inner, dim=1)
        return layer_output

