import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import json

# from core.encoders import *

# from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import json

from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *

from arguments import arg_parse

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from src.utils import (
    AverageMeter,
)
from unsupervised_TU.tsne import *

class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out

class simclr(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, nmb_prototypes=0, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
        self.proj_head = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim))

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(self.embedding_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(self.embedding_dim, nmb_prototypes, bias=False)


        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

        self.args = arg_parse()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, x, edge_index, batch, num_graphs):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)
        # y = F.dropout(y, p=args.dropout_rate, training=self.training)
        # y = self.proj_head(y)

        # if self.l2norm:
        # y = F.normalize(y, dim=1)

        if self.prototypes is not None:
            return y, self.prototypes(y)
        else:
            return y

    def loss_cal(self, x, x_aug, hard_q, neg_q):
        estimator = 'easy'
        temperature = 0.2
        batch_size = x.size(0)

        out_1 = F.normalize(x, dim=1)
        out_2 = F.normalize(x_aug, dim=1)

        # neg score  [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        neg_q = torch.cat([neg_q[0], neg_q[1]], dim=0)
        hard_q = torch.cat([hard_q[0], hard_q[1]], dim=0)
        # neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        sim_matrix  = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()

        # compute distances among prototypes
        w = model.prototypes.weight.data.clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        proto_dist = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)

        # proto_dist = F.normalize(proto_dist, dim=1)

        # negative samples selection
        if args.hard_selection:
            for i, row in enumerate(mask):
                for j, col in enumerate(row):
                    if hard_q[j] not in neg_q[i]:
                        mask[i][j] = False

        if args.sample_reweighting:
            reweight = torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)
            for i, row in enumerate(sim_matrix):
                for j, col in enumerate(row):
                    if i != j:
                        # obtain the prototype id
                        q_i, q_j = hard_q[i].item(), hard_q[j].item()
                        reweight[i][j] = proto_dist[q_i][q_j]
                reweight[i][i] = torch.min(reweight[i])
                # MaxMin scaler
                r_min, r_max = torch.min(reweight[i]), torch.max(reweight[i])
                reweight[i] = (reweight[i] - r_min) / (r_max - r_min)
            # print("before:{}".format(reweight))

            mu = torch.mean(reweight, dim=1)
            std = torch.std(reweight, dim=1)
            # reweight = (reweight - mu) / std
            reweight = torch.exp(torch.pow((reweight - mu), 2) / (2 * torch.pow(std, 2)))
            # print("after:{}".format(reweight))
            sim_matrix = sim_matrix * reweight

        sim_matrix  = sim_matrix.masked_select(mask)#.view(2 * batch_size, -1)
        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos = torch.cat([pos, pos], dim=0)

        loss = (- torch.log(pos / sim_matrix.sum(dim=-1))).mean()


        return loss


import random
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    # dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        # dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

if __name__ == '__main__':

    args = arg_parse()
    setup_seed(args.seed)

    accuracies = {'val':[], 'test':[]}
    epochs = args.epochs
    log_interval = 1
    vis_interval = 1
    batch_size = 128
    # batch_size = 512
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    # args.nmb_prototypes = 10
    print("args.nmb_prototypes:{}".format(args.nmb_prototypes))

    dataset = TUDataset(path, name=DS, aug=args.aug,
                        stro_aug=args.stro_aug).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none',
                             stro_aug='none').shuffle()
    print(len(dataset))
    print(dataset.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = simclr(args.hidden_dim, args.num_gc_layers, nmb_prototypes=args.nmb_prototypes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('epochs: {}'.format(epochs))
    print('================')

    model.eval()
    init_emb, y = model.encoder.get_embeddings(dataloader_eval)
    # print(emb.shape, y.shape)

    """
    acc_val, acc = evaluate_embedding(emb, y)
    accuracies['val'].append(acc_val)
    accuracies['test'].append(acc)
    """

    # build the queue
    queue = None
    args.queue_length -= args.queue_length % (batch_size)

    for epoch in range(1, epochs+1):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        loss_all = 0
        model.train()
        use_the_queue = True
        end = time.time()

        # optionally starts a queue
        # if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
        queue = torch.zeros(
            len(args.crops_for_assign),
            args.queue_length,
            args.hidden_dim * args.num_gc_layers,
            ).cuda()

        global_emb, global_output, global_prot, global_y, global_plabel = [], [], [], [], []

        for it, data in enumerate(dataloader):
            # measure data loading time
            data_time.update(time.time() - end)
            data, data_aug, data_stro_aug = data
            optimizer.zero_grad()
            node_num, _ = data.x.size()
            data = data.to(device)

            bs = data.y.size(0)

            # update learning rate
            iteration = epoch * len(dataloader) + it
            # for param_group in optimizer.param_groups:
            #     param_group["lr"] = lr_schedule[iteration]

            if it == 0:
                global_prot.append(model.prototypes.weight)

            # normalize the prototypes
            with torch.no_grad():
                w = model.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                # print("w:{}".format(w))
                model.prototypes.weight.copy_(w)


            # ============ forward passes ... ============
            # feature, scores
            embedding, output = model(data.x, data.edge_index, data.batch, data.num_graphs)

            global_emb.append(embedding)
            global_output.append(output)
            global_y.append(data.y)

            # print(model.prototypes.weight.size())

            if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == \
                    'ppr_aug' or args.aug == 'random2' or args.aug == 'random3' \
                    or args.aug == 'random4' or args.aug == 'dedge_nodes':
                # node_num_aug, _ = data_aug.x.size()
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)


            if args.stro_aug == 'stro_dnodes' or args.stro_aug == \
                    'stro_subgraph' or args.stro_aug \
                    == 'random2' or args.stro_aug == 'random3' or args.stro_aug == 'random4':
                edge_idx = data_stro_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_stro_aug.x = data_stro_aug.x[idx_not_missing]

                data_stro_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                data_stro_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            data_aug = data_aug.to(device)
            data_stro_aug = data_stro_aug.to(device)

            _embedding, _output = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)

            embedding = torch.cat((embedding, _embedding))
            output = torch.cat((output, _output))
            # embedding = embedding.detach()
            # ============ clustering consistency loss ... ============
            loss = 0
            hard_q, neg_q, z = [], [], []
            for i, crop_id in enumerate(args.crops_for_assign):
                with torch.no_grad():
                    out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                    # time to use the queue
                    if queue is not None:
                        if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                            # print("queue is not None")
                            use_the_queue = True
                            out = torch.cat((torch.mm(
                                queue[i],
                                model.prototypes.weight.t()
                            ), out))
                        # fill the queue
                        queue[i, bs:] = queue[i, :-bs].clone()
                        queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                    # get assignments
                    q = distributed_sinkhorn(out)[-bs:]
                    # if not i:
                    hard_q.append(torch.argmax(q, dim=1))
                    ##################### important hyperparameter #####################
                    neg_q.append(torch.argsort(q, dim=1)[:, 2:8]) # optional choices
                    if not i:
                        global_plabel.append(hard_q[0])
                # cluster assignment prediction
                subloss = 0
                for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                    x = output[bs * v: bs * (v + 1)] / args.temperature
                    z.append(embedding[bs * v: bs * (v + 1), :] / args.temperature)
                    subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))

            contrast_loss = model.loss_cal(z[0], z[1], hard_q, neg_q)
            loss += contrast_loss + 6 * subloss / (np.sum(args.nmb_crops) - 1)


            loss_all += loss.item() #* data.num_graphs
            loss.backward()

            optimizer.step()
            # ============ misc ... ============
            losses.update(loss.item(), data.y.size(0))
            batch_time.update(time.time() - end)
            end = time.time()



        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(emb, y)

        print('Epoch {}, Loss {}, acc {}'.format(epoch, loss_all / len(dataloader),acc))
        accuracies['test'].append(acc)

    print('[info] MAX acc:{}'.format(max(accuracies['test'])))

    tpe  = ('local' if args.local else '') + ('prior' if args.prior else '')
    with open('logs/log_' + args.DS + '_' + args.aug, 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{},{}\n'.format(args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr, s))
        f.write('\n')
