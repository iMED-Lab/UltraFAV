import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

softmax_helper = lambda x: F.softmax(x, 1)
sigmoid_helper = lambda x: F.sigmoid(x)


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or \
                isinstance(module, nn.ConvTranspose2d) or \
                isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class MultiDepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, r=16, L=32, stride=1, is_n8_head=False):
        super(MultiDepthwiseSeparableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_n8_head = is_n8_head
        self.depthwise_convs = nn.ModuleList()
        self.pointwise_convs = nn.ModuleList()
        self.M = len(kernel_sizes)
        d = max(int(out_channels / r), L)

        for kernel_size in kernel_sizes:
            depthwise_conv = nn.Conv2d(in_channels,
                                       in_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=kernel_size // 2,
                                       groups=in_channels)
            self.depthwise_convs.append(depthwise_conv)
            pointwise_conv = nn.Conv2d(in_channels, out_channels, 1)
            self.pointwise_convs.append(pointwise_conv)

        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(out_channels, d, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(d, out_channels * self.M, bias=False))
        self.softmax = nn.Softmax(dim=1)

        self.n8_head = N8Head(in_dim=out_channels, out_dim=8)

        if self.is_n8_head:
            self.squeeze = nn.Sequential(
                nn.Conv2d(out_channels + 8, out_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.squeeze = nn.Identity()

    def forward(self, x):
        feats = []
        bs, c, h, w = x.shape
        for depthwise_conv, pointwise_conv in zip(self.depthwise_convs, self.pointwise_convs):
            x_ = depthwise_conv(x)
            x_ = pointwise_conv(x_)
            x_ = self.norm(x_)
            x_ = self.relu(x_)
            feats.append(x_)
        feats = torch.cat(feats, dim=1)
        feats = feats.view(bs, self.M, self.out_channels, h, w)
        feats_u = torch.sum(feats, dim=1)
        feats_s = self.gap(feats_u).view(bs, self.out_channels)
        feats_z = self.fc(feats_s)
        selector = feats_z.view(bs, self.M, self.out_channels, 1, 1)
        selector = self.softmax(selector)
        feats_fused = torch.sum(feats * selector, dim=1)

        if self.is_n8_head:
            n8 = self.n8_head(feats_fused)
            out = self.squeeze(torch.cat([feats_fused, n8.permute(0, 2, 1).view(bs, 8, h, w)], dim=1))
            return out, n8
        else:
            out = self.squeeze(feats_fused)
            return out


class RefineNet(nn.Module):
    MAX_FILTERS = 480
    BASE__FILTERS = 32

    def __init__(self,
                 input_channels,
                 num_classes,
                 num_pool,
                 deep_supervision=True,
                 is_training=True):
        super(RefineNet, self).__init__()
        self.max_num_features = self.MAX_FILTERS
        self.base_num_features = self.BASE__FILTERS
        self.kernel_sizes = [3, 5, 7]
        self.deep_supervision = deep_supervision
        self.is_training = is_training

        self.encoder = []
        self.decoder = []
        self.down_layers = []
        self.up_layers = []
        self.seg_outputs = []
        self.n8_heads = []

        output_features = self.base_num_features
        input_features = input_channels

        self.filters_per_stage = []
        for d in range(num_pool):
            self.encoder.append(
                MultiDepthwiseSeparableConv2d(
                    in_channels=input_features,
                    out_channels=output_features,
                    kernel_sizes=self.kernel_sizes,
                    is_n8_head=False
                )
            )
            self.down_layers.append(nn.MaxPool2d(2, 2))
            self.filters_per_stage.append(output_features)
            input_features = output_features
            output_features = min(output_features * 2, self.max_num_features)

        final_features = output_features
        self.bottleneck = MultiDepthwiseSeparableConv2d(input_features, final_features, self.kernel_sizes,
                                                        is_n8_head=False)

        self.filters_per_stage = self.filters_per_stage[::-1]
        for u in range(num_pool):
            nfeatures_from_down = final_features
            nfeatures_from_skip = self.filters_per_stage[u]
            nfeatures_after_concat = nfeatures_from_skip * 2

            self.up_layers.append(
                nn.ConvTranspose2d(
                    nfeatures_from_down,
                    nfeatures_from_skip,
                    kernel_size=2,
                    stride=2
                )
            )
            self.decoder.append(
                MultiDepthwiseSeparableConv2d(
                    in_channels=nfeatures_after_concat,
                    out_channels=nfeatures_from_skip,
                    kernel_sizes=self.kernel_sizes,
                    is_n8_head=True
                )
            )
            final_features = nfeatures_from_skip

            self.seg_outputs.append(
                nn.Conv2d(final_features, num_classes, 1, 1, 0, 1, 1)
            )

        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)
        self.down_layers = nn.ModuleList(self.down_layers)
        self.up_layers = nn.ModuleList(self.up_layers)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)

        self.apply(InitWeights_He(1e-2))

    def forward(self, x):
        skips = []
        seg_outputs = []
        n8_outputs = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            skips.append(x)
            x = self.down_layers[i](x)

        x = self.bottleneck(x)

        for i in range(len(self.up_layers)):
            x = self.up_layers[i](x)
            x = torch.cat((x, skips[-(i + 1)]), dim=1)
            x, n8 = self.decoder[i](x)
            seg_outputs.append(softmax_helper(self.seg_outputs[i](x)))
            n8_outputs.append(sigmoid_helper(n8))

        seg_outputs = seg_outputs[::-1]
        n8_outputs = n8_outputs[::-1]
        if self.deep_supervision:
            if self.is_training:
                return seg_outputs, n8_outputs
            else:
                return seg_outputs
        else:
            if self.is_training:
                return seg_outputs[0], n8_outputs[0]
            else:
                return seg_outputs[0]


class N8Head(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 norm_layer=nn.InstanceNorm2d,
                 activation_layer=nn.LeakyReLU):
        """
        out_dim represents the 8-neighbors
        """
        super(N8Head, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = MLP(in_dim=in_dim, out_dim=out_dim, hidden_dim=128)

    def forward(self, x):
        x = self.mlp(x)  # [B, 8, H, W]
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class NCRNet(nn.Module):
    def __init__(self,
                 coarse_model,
                 in_channels,
                 out_channels,
                 num_pool,
                 deep_supervision=True,
                 is_training=True):
        super(NCRNet, self).__init__()
        self.deep_supervision = deep_supervision
        self.is_training = is_training
        self.coarse_model = coarse_model
        self.refine_model = RefineNet(
            input_channels=in_channels + out_channels,
            num_classes=out_channels,
            num_pool=num_pool,
            deep_supervision=deep_supervision,
            is_training=self.is_training
        )

    def forward(self, x):
        coarse_seg = self.coarse_model(x)
        mask = coarse_seg[0] if isinstance(coarse_seg, (list, tuple)) else coarse_seg
        inp = torch.cat((x, mask), dim=1)
        results = self.refine_model(inp)
        if self.is_training:
            refine_seg, n8_outputs = results
            return coarse_seg, refine_seg, n8_outputs
        else:
            refine_seg = results
            return refine_seg


if __name__ == '__main__':
    x = torch.randn(1, 1, 1024, 1536)
    is_training = True
    model = RefineNet(1, 3, 6, deep_supervision=True, is_training=is_training)
    y = model(x)
    if is_training:
        seg, n8 = y
        for i in range(len(seg)):
            print(seg[i].shape, n8[i].shape)
    else:
        seg = y
        for i in range(len(seg)):
            print(seg[i].shape)
