import torch.nn as nn
from .modules import ResNet_FeatureExtractor, BidirectionalLSTM

class CustomModel(nn.Module):

    #def __init__(self, input_channel, output_channel, hidden_size, num_class):
    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        # 로그용
        print(f'custom/models/model.py CustomModel __init__ input_channel : {input_channel}')
        # print(f'custom/models/model.py CustomModel __init__  output_channel : {output_channel}')
        # print(f'custom/models/model.py CustomModel __init__  hidden_size : {hidden_size}')
        # print(f'custom/models/model.py CustomModel __init__  num_class : {num_class}')


        super(CustomModel, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)

        # 로그용
        # print(
        #     f'custom/models/model.py CustomModel __init__  input_channel : {input_channel}')
        # print(
        #     f'custom/models/model.py CustomModel __init__  output_channel : {output_channel}')
        # print(f'custom/models/model.py CustomModel __init__  self.FeatureExtraction : {self.FeatureExtraction}')

        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512

        # 로그용
        #print(f'custom/models/model.py CustomModel __init__  self.FeatureExtraction_output : {self.FeatureExtraction_output}')

        #self.FeatureExtraction_output = output_channel * 4
        #hidden_size = hidden_size * 2
        #self.hidden_size = [int(hidden_size * 2), int(hidden_size / 2)]

        # 로그용
        # print(
        #     f'custom/models/model.py CustomModel __init__  self.FeatureExtraction_output * 2 변경후 BidirectionalLSTM 입력 : {self.FeatureExtraction_output}')
        # print(
        #     f'custom/models/model.py CustomModel __init__  self.FeatureExtraction_output * 2 변경후 BidirectionalLSTM hidden_size : {hidden_size}')

        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        # 로그용
        # print(
        #     f'custom/models/model.py CustomModel __init__  self.SequenceModeling 전 self.FeatureExtraction_output: {self.FeatureExtraction_output}')
        # print(
        #     f'custom/models/model.py CustomModel __init__  self.SequenceModeling 전 hidden_size: {hidden_size}')


        #hidden_size = 128
        #(self, input_size, hidden_size, output_size)
        # 원본
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))

        # self.SequenceModeling = nn.Sequential(
        #     BidirectionalLSTM(self.FeatureExtraction_output, self.hidden_size[0], self.hidden_size[0]),
        #     BidirectionalLSTM(self.hidden_size[0], self.hidden_size[1], self.hidden_size[1]))
        # weight_ih_l0_256 = BidirectionalLSTM(self.FeatureExtraction_output, self.hidden_size[0], self.hidden_size[0])
        # weight_ih_l0_128 = BidirectionalLSTM(self.FeatureExtraction_output, self.hidden_size[1], self.hidden_size[1])
        #weight_ih_l0 = BidirectionalLSTM(self.FeatureExtraction_output, self.hidden_size[1], self.hidden_size[1])

        # 로그용
        # print(
        #     f'custom/models/model.py CustomModel __init__  self.FeatureExtraction_output : {self.FeatureExtraction_output}')

        # print(
        #     f'custom/models/model.py CustomModel __init__  self.SequenceModeling 전 hidden_size: {hidden_size}')

        # self.SequenceModeling = nn.Sequential(
        #     BidirectionalLSTM(self.FeatureExtraction_output, self.hidden_size[0], self.hidden_size[0]),
        #     BidirectionalLSTM(self.hidden_size[0], self.hidden_size[0], self.hidden_size[0]))
        # self.SequenceModeling = nn.Sequential(
        #     BidirectionalLSTM(self.FeatureExtraction_output, self.hidden_size[0], self.hidden_size[0]),
        #     BidirectionalLSTM(self.hidden_size[0], self.hidden_size[0], self.hidden_size[0]))
        # self.SequenceModeling = nn.Sequential(
        #     BidirectionalLSTM(self.FeatureExtraction_output, self.hidden_size[0], self.hidden_size[0]),
        #     BidirectionalLSTM(self.hidden_size[0], self.hidden_size[0], self.hidden_size[0]))

        # 로그용
        # print(
        #     f'custom/models/model.py CustomModel __init__  self.SequenceModeling 전 hidden_size: {hidden_size}')
        #
        # print(
        #     f'custom/models/model.py CustomModel __init__  self.SequenceModeling : {self.SequenceModeling}')

        self.SequenceModeling_output = hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)


    def forward(self, input, text):
        # 로그용
        #print(f'custom/models/model.py CustomModel forward input : {input}')

        # 로그용
        #print(f'custom/models/model.py CustomModel forward input.shape : {input.shape}')
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)
        #print(f'custom/models/model.py CustomModel forward contextual_feature : {contextual_feature}')

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())
        #print(f'custom/models/model.py CustomModel forward prediction : {prediction}')

        return prediction
