import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torchvision import models
from collections import namedtuple
from packaging import version


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class vgg16_bn(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        if version.parse(torchvision.__version__) >= version.parse('0.13'):
            vgg_pretrained_features = models.vgg16_bn(
                weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None
            ).features
        else: #torchvision.__version__ < 0.13
            models.vgg.model_urls['vgg16_bn'] = models.vgg.model_urls['vgg16_bn'].replace('https://', 'http://')
            vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):         # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):         # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):         # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):         # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.Conv2d(1024, 1024, kernel_size=1)
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())        # no pretrained model for fc6 and fc7

        if freeze:
            for param in self.slice1.parameters():      # only first conv
                param.requires_grad= False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        # 로그용
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py BidirectionalLSTM __init__ input_size : {input_size}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py  BidirectionalLSTM __init__ hidden_size : {hidden_size}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py  BidirectionalLSTM __init__ output_size : {output_size}')

        # input_size = input_size / 4
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py BidirectionalLSTM __init__ input_size 다시 4배로 줄인다 : {input_size}')


        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py  BidirectionalLSTM __init__ self.rnn : {self.rnn}')

        self.linear = nn.Linear(hidden_size * 2, output_size)
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py  BidirectionalLSTM __init__ self.linear : {self.linear}')

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        try: # multi gpu needs this
            self.rnn.flatten_parameters()
        except: # quantization doesn't work with this 
            pass
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

class VGG_FeatureExtractor(nn.Module):

    def __init__(self, input_channel, output_channel=256):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))

    def forward(self, input):
        return self.ConvNet(input)

class ResNet_FeatureExtractor(nn.Module):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

    def __init__(self, input_channel, output_channel=512):
        # 로그용
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py custom ResNet_FeatureExtractor  input_channel: {input_channel}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py custom ResNet_FeatureExtractor  output_channel: {output_channel}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py custom ResNet_FeatureExtractor  BasicBlock: {BasicBlock}')

        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def forward(self, input):
        return self.ConvNet(input)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        # 로그용
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py modules BasicBlock __init__ inplanes, : {inplanes,}')
        #
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py modules BasicBlock __init__ planes: {planes}')


        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"

        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py modules BasicBlock _conv3x3 in_planes, : {in_planes,}')

        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py modules BasicBlock _conv3x3 out_planes: {out_planes}')
        t = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py modules BasicBlock _conv3x3 nn.Conv2d: {t}')

        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def forward(self, x):
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py modules BasicBlock forward x: {x}')
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py modules BasicBlock forward self: {self}')

        residual = x

        out = self.conv1(x)
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py modules BasicBlock forward self.conv1(x) out =: {out }')

        out = self.bn1(out)
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py modules BasicBlock forward self.bn1((out)) out =: {out}')

        out = self.relu(out)
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py modules BasicBlock forward self.relu(()) out =: {out}')

        out = self.conv2(out)
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py  BasicBlock forward self.conv2 ) out =: {out}')

        out = self.bn2(out)
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py  BasicBlock forward self.bn2(  out =: {out}')

        if self.downsample is not None:
            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py  BasicBlock forward self.downsample =: {self.downsample}')

            residual = self.downsample(x)
            #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py  BasicBlock forward residual =: {residual}')

        out += residual
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py  BasicBlock forward out =: {out}')

        out = self.relu(out)
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py  BasicBlock forward self.relu(out) out: {out}')

        return out

class ResNet(nn.Module):

    def __init__(self, input_channel, output_channel, block, layers):

        # 로그용
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py custom ResNet __init__ input_channel : {input_channel}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py custom ResNet __init__ output_channel : {output_channel}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py custom ResNet __init__ block : {block}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py custom ResNet __init__ layers : {layers}')

        #output_channel = output_channel * 2

        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py custom ResNet __init__ *2 변경후 output_channel : {output_channel}')


        super(ResNet, self).__init__()

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

        # 로그용
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ self.output_channel_block : {self.output_channel_block}')


        self.inplanes = int(output_channel / 8)
        # 로그용
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ self.inplanes : {self.inplanes}')

        # 로그용
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ input_channel : {input_channel}')
        # 로그용
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ int(output_channel / 16) : {int(output_channel / 16)}')
        # 로그용
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ output_channel : {output_channel}')
        # 로그용
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ int(output_channel / 16) : {int(output_channel / 16)}')
        # 원본
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                 kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 32),
        #                          kernel_size=3, stride=1, padding=1, bias=False)
        # 16 -> 8 로 수정했음
        # self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 8),
        #                          kernel_size=3, stride=1, padding=1, bias=False)
        # 원본
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        # 16 -> 8 로 수정했음
        #self.bn0_1 = nn.BatchNorm2d(int(output_channel / 8))
        #원본
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        # 로그용
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ self.conv0_2 : {self.conv0_2}')
        # 로그용
        #print(int(output_channel / 8))
        # self.conv0_2 = nn.Conv2d(int(output_channel / 8), self.inplanes,
        #                          kernel_size=3, stride=1, padding=1, bias=False)
        # 로그용
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ output_channel : {output_channel}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ int(output_channel / 8) : {int(output_channel / 8)}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ self.inplanes : {self.inplanes}')
        in_chanel = int(output_channel / 8) * 2
        # 로그용
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py in_chanel : {in_chanel}')
        # 16 -> 8 로 수정했음
        #self.conv0_2 = nn.Conv2d(int(output_channel / 8), self.inplanes * 2, kernel_size=3, stride=1, padding=1, bias=False)

        #원본
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        # 로그용
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ self.bn0_2 : {self.bn0_2}')
        #self.bn0_2 = nn.BatchNorm2d(self.inplanes * 2)
        self.relu = nn.ReLU(inplace=True)
        # 로그용
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ self.relu : {self.relu}')

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 로그용
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ self.maxpool1 : {self.maxpool1}')

        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        # 로그용
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ self.layer1 = : {self.layer1  }')

        #원본
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[
                               0], kernel_size=3, stride=1, padding=1, bias=False)
        # 로그용
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ self.conv1 = : {self.conv1  }')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py self.output_channel_block[0] : {self.output_channel_block[0]}')
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py self.output_channel_block[0] * 2 : {self.output_channel_block[0] * 2}')
        # self.output_channel_block[0] = 64
        # self.conv1 = nn.Conv2d(self.output_channel_block[0] * 2, self.output_channel_block[
        #     0] * 2, kernel_size=3, stride=1, padding=1, bias=False)
        #원본
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])
        # 로그용
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ self.bn1 = : {self.bn1 }')


        #self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[
                               1], kernel_size=3, stride=1, padding=1, bias=False)
        # 로그용
        #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!modules.py ResNet __init__ self.conv2 = : {self.conv2}')
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[
                               2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                 3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                 3], kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x

import yaml

# yaml 파일 읽기
yaml_path = 'C:/Users/TAMSystech/yjh/ipynb/deep-text-recognition-benchmark/EasyOCR/user_network/custom.yaml'

with open(yaml_path, 'r', encoding='utf-8') as stream:
    config = yaml.safe_load(stream)

    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!내가 만든 모델 opt : {opt}')

    # 모델 초기화
    input_channel = config['network_params']['input_channel']
    output_channel = config['network_params']['output_channel']
    hidden_size = config['network_params']['hidden_size']
    character = config['character_list']
    num_class = len(character)

    output_channel = output_channel *  2
    # 로그용
    #print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train input_channel  : {input_channel}')
    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train output_channel * 2 변경후  : {output_channel}')
    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train hidden_size  : {hidden_size}')
    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train character  : {character}')
    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!train num_class  : {num_class}')

#ResNet.__init__(input_channel, output_channel, hidden_size, num_class)