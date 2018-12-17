import mxnet as mx
import mxnet.ndarray as nd
from . import Function

class AccMetric:
    def __init__(self, nc, ns, nq, margin=0.4):
        self.acc = 0
        self.nc = nc
        self.nq = nq
        self.ns = ns
        self.margin = margin

    def get(self, pred, label):
        embedding = nd.L2Normalization(pred, mode='instance')
        self.acc = 0
        nc = self.nc; ns = self.ns; nq = self.nq; margin = self.margin

        s_embedding = embedding.slice_axis(axis=0, begin=0, end=nc*ns)
        q_embedding = embedding.slice_axis(axis=0, begin=nc*ns, end=None)
        s_cls_data = nd.reshape(s_embedding, (nc, ns, -1))
        q_cls_data = nd.reshape(q_embedding, (nc, nq, -1))

        s_cls_center = nd.mean(s_cls_data, axis=1)
        s_cls_center = nd.L2Normalization(s_cls_center, mode='instance')

        temp = q_embedding.expand_dims(axis=1) * s_cls_center.expand_dims(axis=0)
        data_center_dis = nd.sum(temp, axis=2)
        cur_label = nd.argmax(data_center_dis, axis=1)

        loss = 0
        # Calculating loss
        for i in range(nc):
            temp = data_center_dis[i*nq:(i+1)*nq, i]
            loss += nd.sum(nd.LeakyReLU(margin - temp, act_type='leaky', slope=0.1))

        for i in range(nc):
            self.acc += nd.sum(cur_label[nq*i:nq*(i+1)] == i).asscalar()
        self.acc /= (nc*nq)

        s_embedding = embedding.slice_axis(axis=0, begin=0, end=nc*ns)
        q_embedding = embedding.slice_axis(axis=0, begin=nc*ns, end=None)

        s_cls_data = nd.reshape(s_embedding, (nc, ns, -1))
        q_cls_data = nd.reshape(q_embedding, (nc, nq, -1))

        s_cls_center = nd.mean(s_cls_data, axis=1)
        s_cls_center = nd.L2Normalization(s_cls_center, mode='instance')
        s_center_broadcast = s_cls_center.expand_dims(axis=1)
        s_center_dis = nd.sum(nd.broadcast_mul(q_cls_data, s_center_broadcast),
                                     axis=2)
        temp = nd.LeakyReLU(margin - s_center_dis, act_type='leaky', slope=0.1)
        loss1 = nd.sum(temp)

        return (self.acc, cur_label, loss)
