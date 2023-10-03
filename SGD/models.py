import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np
import os 
import torch.nn.functional as F
from collections import OrderedDict



class five_layerCNN_MAML(nn.Module):
    def __init__(self,args):

        super(five_layerCNN_MAML,self).__init__()

        self.relu = nn.ReLU()
        
        self.BaseCNN = nn.Sequential(OrderedDict([
            ('conv1',nn.Conv2d(1,32,3,1,1,bias=False)),
            ('relu1', nn.ReLU()),
            ('conv2',nn.Conv2d(32,32,3,1,1,bias=False)),
            ('relu2', nn.ReLU()),
            ('conv3',nn.Conv2d(32,32,3,1,1,bias=False)),
            ('relu3', nn.ReLU()),
            ('conv4',nn.Conv2d(32,32,3,1,1,bias=False)),
            ('relu4', nn.ReLU()),
            ('conv5',nn.Conv2d(32,1,3,1,1,bias=False)),
        ]))
    
    def forward(self,query_input):
        
        cnn_query_output = self.BaseCNN(query_input)
        
        return cnn_query_output
    
    def adaptation(self,support_input,weights):

        #print("input shap: ", us_support_input.size(), "weights: ",weights[0].size())
        x = F.conv2d(support_input,weights[0],stride = 1, padding = 1)
        x = F.relu(x)
        
        x = F.conv2d(x,weights[1],stride = 1, padding = 1)
        x = F.relu(x)
        
        x = F.conv2d(x,weights[2],stride = 1, padding = 1)
        x = F.relu(x)
        
        x = F.conv2d(x,weights[3],stride = 1, padding = 1)
        x = F.relu(x)
        
        adapted_output = F.conv2d(x,weights[4],stride = 1, padding = 1)

        return adapted_output



class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
                      nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
                      nn.InstanceNorm2d(out_chans),
                      nn.ReLU(),
                      nn.Dropout2d(drop_prob),
                    )

    def forward(self, input):
        return self.layers(input)

    def adaptation(self,x,weights1,weights2):
        #print(x.shape,weights1.shape,weights2.shape)
        x = F.conv2d(x,weights1,weights2,stride=1,padding=1)
        x = F.instance_norm(x)
        x = F.relu(x)
        return x

class UnetModel(nn.Module):

    def __init__(self, args,in_chans, out_chans, chans, num_pool_layers, drop_prob):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

        dc = DataConsistencyLayer(args.device)

        self.dc = nn.ModuleList([dc])
        

    def forward(self, us_query_input, ksp_query_imgs, ksp_mask_query):
        stack = []
        output = us_query_input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)
        output_latent = output
        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        output = self.conv2(output) 
        fs_out = output + us_query_input
        output = self.dc[0](fs_out,ksp_query_imgs,ksp_mask_query)
        return output
 
    def adaptation(self, us_support_input, weights,ksp_support_imgs,ksp_mask_support):
        stack = []
        output = us_support_input
        #         ch = chans
        for i in range(self.num_pool_layers):
            output = self.down_sample_layers[i].adaptation(output,weights[2*i],weights[(2*i)+1])
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)
        output = self.conv.adaptation(output,weights[2*self.num_pool_layers],weights[(2*self.num_pool_layers)+1])

        for i in range(self.num_pool_layers):
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = self.up_sample_layers[i].adaptation(output,weights[2*(self.num_pool_layers+1+i)],weights[2*(self.num_pool_layers+1+i)+1])

        finalweightindex = (self.num_pool_layers*2)+1

        output = F.conv2d(output,weights[2*finalweightindex],weights[(2*finalweightindex)+1])
        output = F.conv2d(output,weights[(2*finalweightindex)+2],weights[(2*finalweightindex)+3])
        output = F.conv2d(output,weights[(2*finalweightindex)+4],weights[(2*finalweightindex)+5])
        cnn_support_output = output + us_support_input
        fs_support_output = self.dc[0](cnn_support_output,ksp_support_imgs,ksp_mask_support)

        return fs_support_output