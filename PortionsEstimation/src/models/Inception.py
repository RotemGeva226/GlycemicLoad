from tensorflow.python.ops import nn
import torchvision.models as models
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, type: int, num_classes: int, is_pretrained: bool):
        super(Inception, self).__init__()

        inception_models = {
            3: models.inception_v3
        }

        if is_pretrained:
            weights = 'imagenet'

        self.inception = inception_models[type](weights=weights)
        self.inception.aux_logits = False

        in_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(in_features, num_classes)

    def forward(self, input):
        out = self.inception(input).squeeze(dim=1)
        return out



