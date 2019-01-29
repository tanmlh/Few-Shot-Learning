import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pdb import set_trace as breakpoint

import FeatureModel


def L2SquareDist(A, B, average=True):
    # input A must be:  [nB x Na x nC]
    # input B must be:  [nB x Nb x nC]
    # output C will be: [nB x Na x Nb]
    assert(A.dim()==3)
    assert(B.dim()==3)
    assert(A.size(0)==B.size(0) and A.size(2)==B.size(2))
    nB = A.size(0)
    Na = A.size(1)
    Nb = B.size(1)
    nC = A.size(2)

    # AB = A * B = [nB x Na x nC] * [nB x nC x Nb] = [nB x Na x Nb]
    AB = torch.bmm(A, B.transpose(1,2))

    AA = (A * A).sum(dim=2,keepdim=True).view(nB, Na, 1) # [nB x Na x 1]
    BB = (B * B).sum(dim=2,keepdim=True).view(nB, 1, Nb) # [nB x 1 x Nb]
    # l2squaredist = A*A + B*B - 2 * A * B
    dist = AA.expand_as(AB) + BB.expand_as(AB) - 2 * AB
    if average:
        dist = dist / nC

    return dist


class ProtoHead(nn.Module):
    def __init__(self, opt=None):
        super(ProtoHead, self).__init__()
        scale_cls = opt['scale_cls'] if (opt is not None and 'scale_cls' in opt) else 1.0
        self.scale_cls = nn.Parameter(
            torch.FloatTensor(1).fill_(scale_cls), requires_grad=True)

    def forward(self, support_features, support_labels, query_features, query_labels,
                support_labels_one_hot):
        """Recognize novel categories based on the Prototypical Nets approach.

        Classify the test examples (i.e., `features_test`) using the available
        training examples (i.e., `features_test` and `labels_train`) using the
        Prototypical Nets approach.

        Args:
            features_test: A 3D tensor with shape
                [batch_size x num_test_examples x num_channels] that represents
                the test features of each training episode in the batch.
            features_train: A 3D tensor with shape
                [batch_size x num_train_examples x num_channels] that represents
                the train features of each training episode in the batch.
            labels_train: A 3D tensor with shape
                [batch_size x num_train_examples x nKnovel] that represents
                the train labels (encoded as 1-hot vectors) of each training
                episode in the batch.

        Return:
            scores_cls: A 3D tensor with shape
                [batch_size x num_test_examples x nKnovel] that represents the
                classification scores of the test feature vectors for the
                nKnovel novel categories.
        """



        assert support_features.dim() == 3
        assert support_labels_one_hot.dim() == 3
        assert query_features.dim() == 3
        assert support_features.size(0) == support_labels_one_hot.size(0)
        assert support_features.size(0) == query_features.size(0)
        assert support_features.size(1) == support_labels_one_hot.size(1)
        assert support_features.size(2) == query_features.size(2)

        batch_size = support_features.size()[0]
        num_query = query_features.size()[1]

        #************************* Compute Prototypes **************************
        support_labels_transposed = support_labels_one_hot.transpose(1,2)
        # Batch matrix multiplication:
        #   prototypes = labels_train_transposed * features_train ==>
        #   [batch_size x nKnovel x num_channels] =
        #       [batch_size x nKnovel x num_train_examples] * [batch_size * num_train_examples * num_channels]
        prototypes = torch.bmm(support_labels_transposed, support_features)
        # Divide with the number of examples per novel category.
        prototypes = prototypes.div(
            support_labels_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
        )
        #***********************************************************************
        scores_cls2 = -self.scale_cls * L2SquareDist(query_features, prototypes)
        scores_cls2 = scores_cls2.view(batch_size*num_query, -1)


        batch_size, num_support, num_features = support_features.size()
        num_query = query_features.size()[1]
        unsqueeze_query = query_features.unsqueeze(2).repeat(1, 1, num_support, 1)
        unsqueeze_support = support_features.unsqueeze(1).repeat(1, num_query, 1, 1)
        raw_relations = -torch.norm(unsqueeze_query - unsqueeze_support, p=2, dim=3).pow(2) / query_features.size(-1)
        relations = torch.bmm(raw_relations, support_labels_one_hot)
        scores_cls = relations.view(batch_size * num_query, -1)

        query_labels = query_labels.view(batch_size*num_query)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(scores_cls, query_labels)

        accuracy = 1.0 * (scores_cls.argmax(dim=1) == query_labels).sum().item() / (batch_size*num_query)

        res = {}
        res['loss'] = loss
        res['accuracy'] = accuracy
        res['pred_labels'] = list(scores_cls.argmax(dim=1).cpu().numpy())
        res['gt_labels'] = list(query_labels.cpu().numpy())


        return res

def create_model(opt=None):
    return ProtoHead(opt)
