import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, 2)
        loss = 0.5 * (label * euclidean_distance.pow(2) +
                      (1 - label) * F.relu(self.margin - euclidean_distance).pow(2))
        return loss.mean()

class CrossEntropyLossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(CrossEntropyLossWithLabelSmoothing, self).__init__()
        self.smoothing = smoothing

    def forward(self, output, target):
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)
        log_prob = F.log_softmax(output, dim=1)
        return -(one_hot * log_prob).sum(dim=1).mean()
