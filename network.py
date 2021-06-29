import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


def init_weight(net, init_type, gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if (hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1)):
            if (init_type == "normal"):
                init.normal_(m.weight.data, 0.0, gain)
            elif (init_type == "xavier"):
                init.xavier_normal_(m.weight.data, gain)
            elif (init_type == "kaiming"):
                init.kaiming_normal_(m.weight.data, )
            elif (init_type == "orthogonal"):
                init.orthogonal_(m.weight.data, gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type="normal", gain=0.02, gpu_id="cude:0"):
    net.to(gpu_id)
    init_weight(net, init_type, gain)
    return net


def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_id='cuda:0'):
    net = None
    norm_layer = get_norm_layer(norm)
    net = ResnetGenerator(input_nc, output_nc, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_block=4)
    return init_net(net, init_type=init_type, gain=init_gain, gpu_id=gpu_id)


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, gpu_id)


def get_norm_layer(norm_type):
    if (norm_type == 'batch'):
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif (norm_type == 'instance'):
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    return norm_layer


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_block=9,
                 padding_type='reflect'):
        assert (n_block >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if (type(norm_layer) == functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.inc = Inconv(input_nc, ngf, norm_layer, use_bias)
        self.down1 = Down(ngf, ngf * 2, norm_layer, use_bias)
        self.down2 = Down(ngf * 2, ngf * 4, norm_layer, use_bias)
        # self.down3 = Down(ngf * 4, ngf * 4, norm_layer, use_bias)
        model = []
        for i in range(n_block):
            model.append(ResBlock(ngf * 4, padding_type, norm_layer, use_dropout, use_bias))
        self.resblocks = nn.Sequential(*model)
        self.up1 = Up(ngf * 4, ngf * 2, norm_layer, use_bias)
        self.up2 = Up(ngf * 2, ngf, norm_layer, use_bias)
        # self.up3 = Up(ngf * 2, ngf, norm_layer, use_bias)
        self.outc = Outconv(ngf, output_nc)

    def forward(self, x):
        out = {}
        out["in"] = self.inc(x)
        out["d1"] = self.down1(out["in"])
        out["d2"] = self.down2(out["d1"])
        # out["d3"] = self.down3(out["d2"])
        # print(out["d3"].shape)
        out["bottle"] = self.resblocks(out["d2"])
        out["u1"] = self.up1(out["bottle"])
        out["u2"] = self.up2(out["u1"])
        # out["u3"] = self.up3(out["u2"])
        return self.outc(out["u2"])

    def show_network(self):
        for name, module in self.named_children():
            print(name, module)


class Inconv(nn.Module):
    def __init__(self, input_nc, ngf, norm_layer, use_bias):
        super(Inconv, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf
        self.norm_layer = norm_layer
        self.use_bias = use_bias
        self.inconv = nn.Sequential(nn.ReflectionPad2d(2),
                                    nn.Conv2d(self.input_nc, self.ngf, kernel_size=7, padding=0, bias=use_bias),
                                    norm_layer(self.ngf),
                                    nn.ReLU(True))

    def forward(self, x):
        return self.inconv(x)


class Down(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer, use_bias):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.norm_layer = norm_layer
        self.use_bias = use_bias
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(self.input_nc, self.output_nc, kernel_size=3, stride=2, padding=1, bias=self.use_bias),
            norm_layer(self.output_nc),
            nn.ReLU(True))

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Up, self).__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(in_ch,
                                                   out_ch,
                                                   kernel_size=3,
                                                   stride=2, padding=1,
                                                   output_padding=1,
                                                   bias=use_bias),
                                norm_layer(out_ch),
                                nn.ReLU(True))

    def forward(self, x):
        return self.up(x)


class ResBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if (padding_type == "reflect"):
            conv_block.append(nn.ReflectionPad2d(1))
        elif (padding_type == "replicate"):
            conv_block.append(nn.ReplicationPad2d(1))
        elif (padding_type == "zero"):
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block.append(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias))
        conv_block.append(norm_layer(dim))
        conv_block.append(nn.ReLU(True))

        if (use_dropout):
            conv_block.append(nn.Dropout2d(0.5))
        p = 0
        if (padding_type == "reflect"):
            conv_block.append(nn.ReflectionPad2d(1))
        elif (padding_type == "replicate"):
            conv_block.append(nn.ReplicationPad2d(1))
        elif (padding_type == "zero"):
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block.append(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias))
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return nn.ReLU(True)(out)


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.outconv = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
                                     nn.Tanh())

    def forward(self, x):
        return self.outconv(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
                     norm_layer(ndf * nf_mult),
                     nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if (use_sigmoid):
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

    def show_network(self):
        for name, module in self.named_children():
            print(name, module)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

    def show_network(self):
        for name, module in self.named_children():
            print(name, module)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if (target_is_real):
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


input_nc = 3
output_nc = 3
ngf = 4
dim = 3
# padding_type = "zero"
norm_layer = get_norm_layer("batch")
use_dropout = True
use_bias = True
# model = ResnetGenerator(input_nc,output_nc,ngf,norm_layer,)

# model = ResBlock(dim,padding_type,norm_layer,use_dropout,use_bias)
# Up = Up(3,6,norm_layer,use_bias)
# Input = torch.rand(50, 3, 256, 256)
# output = model(Input)
# print(output.shape)
# print(model.show_network())
# Up_output = Up(Input)
# print(Up_output.shape)

# model = NLayerDiscriminator(input_nc)
# output = model(Input)
# print(output.shape)
# model = define_G(input_nc,output_nc,ngf,gpu_id="cpu")
# model.show_network()
# Output = model(Input)
# print(Output.shape)
# loss = GANLoss(use_lsgan=False)
# model = define_D(input_nc,64,"pixel",use_sigmoid=True,gpu_id="cpu")
# model.show_network()
# output = model(Input)
# l = loss(output,1)
# print(l)
