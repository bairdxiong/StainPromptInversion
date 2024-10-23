import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetSGNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.UpBlock2 = nn.Sequential(*UpBlock2)

    def forward(self, input,label,device):
        x = self.DownBlock(input)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x,label,device)
        out = self.UpBlock2(x)

        return out


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetSGNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetSGNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = SGN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = SGN(dim)

    def forward(self, x, label,device):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, label,device)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, label,device)

        return out+x

class SGN(nn.Module):#Style Guided Normalization (SGN)
    def __init__(self, num_features, eps=1e-5):
        super(SGN, self).__init__()
        self.eps = eps
        self.weight_1 = Parameter(torch.Tensor(1,4,4))
        self.weight_2 = Parameter(torch.Tensor(1,4,1))
        self.gamma_1 = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma_2 = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta_1 = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta_2 = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        torch.nn.init.normal_(self.weight_1, mean=0, std=1)
        self.weight_2.data.fill_(1.0)
        self.rho.data.fill_(0.9)
        self.gamma_1.data.fill_(1.0)
        self.beta_1.data.fill_(1.0)
        self.gamma_2.data.fill_(0.0)
        self.beta_2.data.fill_(0.0)

    def forward(self, input,label,device):
        # While testing, we directly feed our network one-hot labels, so we do not need this construction of improved one-hot label (line 131-135).
        newlabel=torch.zeros((input.shape[0],1,4)).to(device) #The construction of improved one-hot label
        for i in range(4):
            if i==label:
                newlabel[:,:,i]=1
            else:newlabel[:,:,i]=0.05 #The value of delta

        for i in range(input.shape[0]):#Multiple Style Encoding (MSE)
            if i ==0:
                newweight=torch.mm(newlabel[i],self.weight_1[i])
                newweight=torch.sigmoid(newweight)
                newweight=torch.mm(newweight,self.weight_2[i])#batch size==1
            else:newweight=torch.cat((newweight,torch.mm(newlabel[i],self.weight[0])),0)#We haven't implemented the cases where batch size!=1, so this code only suits the cases where batch size==1.


        gamma=self.gamma_1*newweight+self.gamma_2
        beta=self.beta_1*newweight+self.beta_2
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.expand(input.shape[0], -1, -1, -1) + beta.expand(input.shape[0], -1, -1, -1)

        return out

class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)

        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))


        self.gap_fc_2 = nn.utils.spectral_norm(nn.Linear(ndf * mult, 4, bias=False))
        self.gmp_fc_2 = nn.utils.spectral_norm(nn.Linear(ndf * mult, 4, bias=False))
        self.leaky_relu_2 = nn.LeakyReLU(0.2, True)
        self.pad_2 = nn.ReflectionPad2d(1)
        for i in range(4):
            setattr(self, 'conv1*1_2_' + str(i), nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True))
        for i in range(4):
            setattr(self, 'conv_2_' + str(i), nn.utils.spectral_norm(nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False)))

        self.model = nn.Sequential(*model)

    def forward(self, input,device):
        x0 = self.model(input)
        heatmap_0 = torch.sum(x0, dim=1, keepdim=True)

        #Discrimination Module and Auxiliary Classifier 2
        x=x0
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap_1 = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)


        #Classification Module and Auxiliary Classifier 1
        out_2=torch.zeros((input.shape[0],4),dtype=float)
        x=x0

        gap_2 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit_2 = self.gap_fc_2(gap_2.view(x.shape[0], -1)).unsqueeze(2)

        gmp_2 = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit_2 = self.gmp_fc_2(gmp_2.view(x.shape[0], -1)).unsqueeze(2)

        cam_logit_2_0 = torch.cat([gap_logit_2, gmp_logit_2], 2)
        cam_logit_2 = torch.mean(cam_logit_2_0,dim=2,keepdim=False)
        Softmax = nn.Softmax(dim=1)
        cam_logit_2=Softmax(cam_logit_2)

        heatmap_2=torch.Tensor(x.shape[0],0,x.shape[2],x.shape[3]).to(device)

        for i in range(4):
            x = x0
            gap_weight_2 = list(self.gap_fc_2.parameters())[0][i].unsqueeze(0)
            gap_2 = x * gap_weight_2.unsqueeze(2).unsqueeze(3)

            gmp_weight_2 = list(self.gmp_fc_2.parameters())[0][i].unsqueeze(0)
            gmp_2 = x * gmp_weight_2.unsqueeze(2).unsqueeze(3)

            x = torch.cat([gap_2, gmp_2], 1)
            x = self.leaky_relu_2(getattr(self, 'conv1*1_2_' + str(i))(x))

            heatmap_2_0 = torch.sum(x, dim=1, keepdim=True)
            heatmap_2=torch.cat((heatmap_2,heatmap_2_0),1)#attention map

            x = self.pad_2(x)
            out_2_0 = getattr(self, 'conv_2_' + str(i))(x)
            if i==0:
                out_2 = torch.mean(out_2_0,axis=(2,3),keepdim=False)
            else:out_2=torch.cat((out_2,torch.mean(out_2_0,axis=(2,3),keepdim=False)),1)

        Softmax = nn.Softmax(dim=1)
        out_2 = Softmax(out_2)

        return out, cam_logit,out_2,cam_logit_2,heatmap_0,heatmap_1,heatmap_2[:,0],heatmap_2[:,1],heatmap_2[:,2],heatmap_2[:,3]


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w
