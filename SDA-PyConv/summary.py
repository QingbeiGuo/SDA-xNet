#!/usr/bin/python
# -*- coding: UTF-8 -*-

#pytorch -0.2.1
#python -3.6.2
#torchvision - 0.1.9

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import os

from collections import OrderedDict
import numpy as np

#from models.resnet_pyconv.resnet_pyconv import PyConvResNet, pyconvresnet50
from models.resnet_pyconv.resnet_pyconv_at import PyConvResNet, pyconvresnet50

def summary(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            #calculate params
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

            #calculate flops
            flops = 0
            if hasattr(module, 'weight') and module.weight is not None:
                if isinstance(module, nn.Conv2d):
                    _, _, output_height, output_width = output.size()
                    output_channel, input_channel, kernel_height, kernel_width = module.weight.size()
                    flops = output_channel * output_height * output_width * input_channel * kernel_height * kernel_width
                if isinstance(module, nn.Linear):
                    input_num, output_num = module.weight.size()
                    flops = input_num * output_num
                summary[m_key]['weight'] = list(module.weight.size())
            else:
                summary[m_key]['weight'] = 'None'
            if hasattr(module, 'bias') and module.bias is not None:
                flops += module.bias.numel()
            summary[m_key]["flops"] = flops

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("---------------------------------------------------------------------------------------------------------------------------")
    line_new = "{:>20}  {:>25}   {:>25} {:>15} {:>15} {:>15}".format("Layer (type)", "Input Shape", "Output Shape", "Weight", "Param #", "FLOPs #")
    print(line_new)
    print("===========================================================================================================================")
    total_params = 0
    total_flops = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25}  {:>25} {:>15} {:>15} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            str(summary[layer]["weight"]),
            "{0:,}".format(summary[layer]["nb_params"]),
            "{0:,}".format(summary[layer]["flops"]),
        )
        total_params += summary[layer]["nb_params"]
        total_flops += summary[layer]["flops"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("===========================================================================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Total flops: {0:,}".format(total_flops))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("---------------------------------------------------------------------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("---------------------------------------------------------------------------------------------------------------------------")
    # return summary

##############################################################################################################
if __name__ == '__main__':

	os.environ["CUDA_VISIBLE_DEVICES"] = "3"

	model = pyconvresnet50().cpu()
	#model = resnet101_cbam().cpu()
	print("model:", model)

	#summary(model, (3, 32, 32), device="cpu")
	summary(model, (3, 224, 224), device="cpu")
