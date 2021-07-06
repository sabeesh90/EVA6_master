import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet_Mod(nn.Module):
    def __init__(self, model):
        super(ResNet_Mod, self).__init__()
        
        # loading the model
        self.res = model   
        # accessing the last convolutional layer     
        self.last_layer = nn.Sequential(*list(self.res.children())[:-1])     
        # accessing the last classifier layer
        self.classifier = nn.Sequential(*list(self.res.children())[-1:])

        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.last_layer(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        # x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
