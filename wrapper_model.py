import torch
import torch.nn as nn

# Define a wrapper class
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Call the original model
        output = self.model(x)

        # Convert the dict output to a tuple
        return output.get('inpainted')#tuple(output.values())

# Create an instance of the wrapper with your model
wrapped_model = ModelWrapper(model)

# Create your dummy inputs
dummy_image = torch.randn(1, 3, 512, 512).to('cuda')
dummy_mask = torch.randn(1, 1, 512, 512).to('cuda')
dummy_input = {'image': dummy_image, 'mask': dummy_mask}
