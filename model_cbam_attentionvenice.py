import torch.nn as nn
import torch
from torchvision import models
from utils import save_net, load_net
from crossattention_augmented_conv import AugmentedConv

class BMM(nn.Module):
    def __init__(self):
        super(BMM, self).__init__()

    def forward(self, q, k):
        return torch.bmm(q, k)

class SelfAttention(nn.Module):
    " Self attention Layer"

    def __init__(self, in_dim, activation='relu'):
        super(SelfAttention, self).__init__()
        self.channel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.bmm = BMM()
        self.softmax = nn.Softmax(dim=1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = self.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        return out


class PHNet(nn.Module):
    def __init__(self, load_weights=False):
        super(PHNet, self).__init__()
        self.seen = 0
        self.frame = 3
        self.conv11 = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(self.frame + 1, 1, 1), stride=1)
        self.CRPool = nn.Conv3d(3, 3, kernel_size=(2, 3, 3), stride=1, padding=(1, 1, 1), bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = x.clone()
        y = self.CRPool(y)
        y = self.conv11(y * y)
        y = torch.squeeze(y, dim=2)
        return y


class STNNet1(nn.Module):
    def __init__(self, in_channels):
        super(STNNet1, self).__init__()
        # 定位网络-卷积层
        self.localization_convs = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        # 定位网络-线性层
        self.localization_linear = nn.Sequential(
            nn.Linear(in_features=10 * 156 * 86, out_features=32),#应该是需要修改这里

            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2 * 3)
        )
        # 初始化定位网络仿射矩阵的权重/偏置，即是初始化θ值。使得图片的空间转换从原始图像开始。
        self.localization_linear[2].weight.data.zero_()
        self.localization_linear[2].bias.data.copy_(torch.tensor([1, 0, 0,
                                                                  0, 1, 0], dtype=torch.float))

    # 空间变换器网络，转发图片张量
    def stn(self, x):
        # 使用CNN对图像结构定位，生成变换参数矩阵θ（2*3矩阵）
        x2 = self.localization_convs(x)
        print(x2.shape)
        x2 = x2.view(x2.size()[0], -1)
        print(x2.shape)
        x2 = self.localization_linear(x2)
        theta = x2.view(x2.size()[0], 2, 3)  # [1, 2, 3]
        # print(theta)
        '''
        2D空间变换初始θ参数应置为tensor([[[1., 0., 0.],
                                        [0., 1., 0.]]])
        '''
        # 网格生成器，根据θ建立原图片的坐标仿射矩阵
        grid = nn.functional.affine_grid(theta, x.size(), align_corners=True)  # [1, 28, 28, 2]
        # 采样器，根据网格对原图片进行转换，转发给CNN分类网络
        x = nn.functional.grid_sample(x, grid, align_corners=True)  # [1, 1, 28, 28]
        return x

    def forward(self, x):
        x = self.stn(x)
        return x


class STNNet2(nn.Module):
    def __init__(self, in_channels):
        super(STNNet2, self).__init__()
        # 定位网络-卷积层
        self.localization_convs = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        # 定位网络-线性层
        self.localization_linear = nn.Sequential(
            nn.Linear(in_features=128 * 80 * 45, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2 * 3)
        )
        # 初始化定位网络仿射矩阵的权重/偏置，即是初始化θ值。使得图片的空间转换从原始图像开始。
        self.localization_linear[2].weight.data.zero_()
        self.localization_linear[2].bias.data.copy_(torch.tensor([1, 0, 0,
                                                                  0, 1, 0], dtype=torch.float))

        # # 图片分类-卷积层
        # self.convs = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=4),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3),
        # )
        # # 图片分类-线性层
        # self.linear = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(in_features=512, out_features=10),
        # )

    # 空间变换器网络，转发图片张量
    def stn(self, x):
        # 使用CNN对图像结构定位，生成变换参数矩阵θ（2*3矩阵）
        x2 = self.localization_convs(x)

        x2 = x2.view(x2.size()[0], -1)
        x2 = self.localization_linear(x2)
        theta = x2.view(x2.size()[0], 2, 3)  # [1, 2, 3]
        # print(theta)
        '''
        2D空间变换初始θ参数应置为tensor([[[1., 0., 0.],
                                        [0., 1., 0.]]])
        '''
        # 网格生成器，根据θ建立原图片的坐标仿射矩阵
        grid = nn.functional.affine_grid(theta, x.size(), align_corners=True)  # [1, 28, 28, 2]
        # 采样器，根据网格对原图片进行转换，转发给CNN分类网络
        x = nn.functional.grid_sample(x, grid, align_corners=True)  # [1, 1, 28, 28]
        return x

    def forward(self, x):
        x = self.stn(x)
        return x


# 加入注意力机制（CBAM）
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class Layer_out1(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size, d_rate):
        super(Layer_out1, self).__init__()
        self.d_rate = 2
        self.function = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size, padding=d_rate, dilation=d_rate),
            cbam_block(256),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )
        self.downsample = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size, padding=d_rate, dilation=d_rate)
        )

    def forward(self, x):
        identify = x

        identify = self.downsample(identify)
        # print(identify.shape)
        f = self.function(x)
        # print(f.shape)
        x1 = f + identify
        return x1


class Layer_out2(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size, d_rate):
        super(Layer_out2, self).__init__()
        self.function = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size, padding=d_rate, dilation=d_rate),
            cbam_block(128),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )
        self.downsample = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size, padding=d_rate, dilation=d_rate)
        )

    def forward(self, x):
        identify = x
        identify = self.downsample(identify)
        f = self.function(x)
        x2 = f + identify
        return x2


class Layer_out3(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size, d_rate):
        super(Layer_out3, self).__init__()
        self.function = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size, padding=d_rate, dilation=d_rate),
            cbam_block(64),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )
        self.downsample = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size, padding=d_rate, dilation=d_rate)
        )

    def forward(self, x):
        identify = x
        identify = self.downsample(identify)
        f = self.function(x)
        x3 = f + identify
        return x3


class NewCSRNet(nn.Module):
    def __init__(self, load_weights=False):  # self指的是调用该函数的对象
        super(NewCSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # self.backend_feat = [128, 128, 64]
        self.backend_feat = [128,128, 64]
        self.frontend = make_layers_front(self.frontend_feat)
        self.layer_out1 = Layer_out1(512, 256, 3, 2)
        self.layer_out2 = Layer_out2(256, 128, 3, 2)
        self.layer_out3 = Layer_out3(128, 64, 3, 2)
        # self.backend = make_layers(self.backend_feat, in_channels=128, batch_norm=True, dilation=True)
        self.backend = make_layers_front(self.backend_feat, in_channels=128, batch_norm=True, dilation=True)
        self.stn1 = STNNet1(3)
        self.ph = PHNet()
        self.stn2 = STNNet2(128)
        self.relu = nn.ReLU()
        self.xatt = AugmentedConv(64, 64, kernel_size=3, dk=64, dv=32, Nh=1)
        self.aggreg = None
        # self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)

        self.output_layer = nn.Conv2d(64, 10, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    # def forward(self,x_prev,x):
    #     x_prev = self.frontend(x_prev)
    #     x = self.frontend(x)
    #
    #     x_prev = self.context(x_prev)
    #     print(x_prev.shape)
    #     x = self.context(x)
    #     print(x.shape)
    #
    #
    #     x = torch.cat((x_prev,x),1)
    #     print(x.shape)
    #
    #     x = self.backend(x)
    #     x = self.output_layer(x)
    #     x = self.relu(x)
    #     return x

    def forward(self, x_perv, x):

        x_perv = self.stn1(x_perv)
        x = self.stn1(x)
        x_perv = self.frontend(x_perv)
        x = self.frontend(x)
        # print(x.shape)
        x_perv = self.layer_out1(x_perv)
        x = self.layer_out1(x)
        x_perv = self.layer_out2(x_perv)
        x = self.layer_out2(x)
        x_perv = self.layer_out3(x_perv)
        x = self.layer_out3(x)
        print(x.shape)

        xatt1, weights1 = self.xatt(x, x_perv)
        xatt2, weights2 = self.xatt(x_perv, x)
        # x_perv = self.output_layer(x_perv)
        # if self.aggreg is None:
        #     x = torch.cat((xatt1, xatt2), 1)
        # else:
        #     x = self.aggreg(xatt1, xatt2)
        x = torch.cat((xatt1, xatt2), 1)
        # x = torch.cat((x_perv, x), 1)

        x = self.backend(x)

        x = self.output_layer(x)
        # x = self.stn2(x)
        x = self.relu(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers_front(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:

            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            channelattention1 = cbam_block(channel=v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, channelattention1, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, channelattention1, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def zero(m):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.zeros_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)


if __name__ == '__main__':
    model = NewCSRNet(load_weights=True)
    # print("model's state_dict")
    # for param_tensor in model.state_dict():
    #     print(param_tensor,"\t",model.state_dict()[param_tensor].size())
    # torch.save(model,'./a')
    # model = torch.load('./a')
    # print(model)
    # model.train()
    # x = torch.rand(3, 3, 1280, 720)
    # x_prev = torch.rand(3, 3, 1280, 720)
    x = torch.rand(3, 3, 640, 480)
    x_prev = torch.rand(3, 3, 640, 480)
    print(model(x, x_prev).shape)
    #print(model)


