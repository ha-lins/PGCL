import argparse

from loader import MoleculeDataset_aug
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_Virtualnode
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from tensorboardX import SummaryWriter

from copy import deepcopy
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from logging import getLogger
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import apex
from apex.parallel.LARC import LARC
import math

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module):
    def __init__(self, emb_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)


class graphcl(nn.Module):

    def __init__(self, gnn, emb_dim, num_layer, nmb_prototypes=0, alpha=0.5, beta=1., gamma=.1):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.embedding_dim = emb_dim #* num_layer
        self.projection_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim))
        # prototype layer
        self.prototypes = None

        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(self.embedding_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(self.embedding_dim, nmb_prototypes, bias=False)

    def forward_cl(self, x, edge_index, edge_attr, batch):
        # x = self.gnn(x, edge_index, edge_attr)
        x = self.gnn(x, edge_index, edge_attr, batch)
        x = self.pool(x, batch)
        x = self.projection_head(x)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        else:
            return x

    # def loss_cl(self, x1, x2):
    #     T = 0.1
    #     batch_size, _ = x1.size()
    #     x1_abs = x1.norm(dim=1)
    #     x2_abs = x2.norm(dim=1)
    #
    #     sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    #     sim_matrix = torch.exp(sim_matrix / T)
    #     pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    #     loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    #     loss = - torch.log(loss).mean()
    #     return loss

    def loss_cal(self, x, x_aug, hard_q, neg_q, args):
        estimator = 'easy'
        temperature = 0.2
        batch_size = x.size(0)
        # print("x:{}".format(x.size()))

        # out_1 = x.norm(dim=1, keepdim=True)
        # out_2 = x_aug.norm(dim=1, keepdim=True)

        out_1 = F.normalize(x, dim=1)
        out_2 = F.normalize(x_aug, dim=1)

        # neg score  [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        neg_q = torch.cat([neg_q[0], neg_q[1]], dim=0)
        hard_q = torch.cat([hard_q[0], hard_q[1]], dim=0)
        # print("hard_q:{}".format(hard_q))
        # neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        sim_matrix  = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).type(torch.bool)

        # compute distances among prototypes
        w = self.prototypes.weight.data.clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        proto_dist = 1 - torch.mm(w, w.t().contiguous())#F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
        sample_dist = 1 - torch.mm(out, out.t().contiguous())#F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=2)

        # negative samples selection
        if args.hard_selection:
            for i, row in enumerate(mask):
                for j, col in enumerate(row):
                    if hard_q[j] not in neg_q[i]:
                        mask[i][j] = False

        # reweighting with prototype distance
        if args.proto_dist:
            reweight = torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)
            for i, row in enumerate(sim_matrix):
                for j, col in enumerate(row):
                    if i != j:
                        # obtain the prototype id
                        q_i, q_j = hard_q[i].item(), hard_q[j].item()
                        reweight[i][j] = proto_dist[q_i][q_j]

                # MaxMin scaler
                r_min, r_max = torch.min(reweight[i]), torch.max(reweight[i])
                reweight[i] = (reweight[i] - r_min) / (r_max - r_min)

            mu = torch.mean(reweight, dim=1)
            std = torch.std(reweight, dim=1)
            reweight = torch.exp(torch.pow((reweight - mu), 2) / (2 * torch.pow(std, 2)))
            sim_matrix = sim_matrix * reweight

        # reweighting with sample distance
        if args.sample_dist:
            weight_mask = torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)
            reweight = weight_mask * sample_dist
            reweight = nn.functional.normalize(reweight, dim=1, p=2)
            # for i, row in enumerate(reweight):
            # for j, col in enumerate(row):
            #     if i != j:
            #         reweight[i][j] = sample_dist[i][j]

            #MaxMin scaler
            # r_min, r_max = row.min(), row.max()
            # reweight[i] = (row - r_min) / (r_max - r_min)

            mu = torch.mean(reweight, dim=1)
            std = torch.std(reweight, dim=1)
            reweight = torch.exp(torch.pow((reweight - mu), 2) / (2 * torch.pow(std, 2)))
            sim_matrix = sim_matrix * reweight

        sim_matrix  = sim_matrix.masked_select(mask)#.view(2 * batch_size, -1)
        # print(sim_matrix)
        # pos score
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        return loss

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))


import random
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

@torch.no_grad()
def distributed_sinkhorn(out):
    # print("out: ", torch.any(torch.isinf(out)))
    # print("out: ", torch.any(torch.isnan(out)))

    epsilon = 0.05
    world_size = 1
    sinkhorn_iterations = 3
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper

    B = Q.shape[1] * world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    # dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
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

def train(args, model, device, dataset, optimizer, lr_schedule, writer, epoch):

    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)
    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)

    model.train()
    use_the_queue = True

    # optionally starts a queue
    queue = torch.zeros(
        len(args.crops_for_assign),
        args.queue_length,
        args.emb_dim, #* args.num_layer,
    ).cuda()

    global_emb, global_output, global_prot, global_y, global_plabel = [], [], [], [], []

    train_acc_accum = 0
    train_loss_accum = 0
    for step, batch in enumerate(tqdm(zip(loader1, loader2), desc="Iteration")):
        # update learning rate
        iteration = epoch * len(loader1) + step
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        if not step:
            global_prot.append(model.prototypes.weight)

        # normalize the prototypes
        with torch.no_grad():
            w = model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            # print("w:{}".format(w))
            model.prototypes.weight.copy_(w)

        batch1, batch2 = batch
        bs = batch1.id.size(0)
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()

        embedding, output = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        _embedding, _output = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)

        embedding = torch.cat((embedding, _embedding))
        output = torch.cat((output, _output))
        # print("embedding: {}".format(embedding))
        # print("output: {}".format(output))
        # ============ consistency loss ... ============
        loss, contrast_loss, consistency_loss = 0, 0, 0
        hard_q, neg_q, z = [], [], []
        for i, crop_id in enumerate(args.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                if queue is not None:
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        # print("queue is not None")
                        use_the_queue = True
                        # print("model.prototypes.weight: ", torch.any(torch.isnan(model.prototypes.weight)))
                        # print("queue: ", torch.any(torch.isnan(queue)))

                        out = torch.cat((torch.mm(
                            nn.functional.normalize(queue[i], dim=1, p=2),
                            model.prototypes.weight.t()
                        ), out))
                        # print("out: ", torch.any(torch.isnan(out)))
                        # print("queue[i]: ", torch.any(torch.isnan(queue[i])))
                        # print("model.prototypes.weight: ", torch.any(torch.isnan(model.prototypes.weight)))

                        # print("queue[i]:{}".format(queue[i]))
                        # print("model.prototypes.weight.t():{}".format(model.prototypes.weight.t()))
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    # print(queue.size(), embedding.size(), bs)
                    queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                # print(out.size())
                q = distributed_sinkhorn(nn.functional.normalize(out, dim=1, p=2))[-bs:]
                # q = distributed_sinkhorn(out)[-bs:]
                # print("q: ", torch.any(torch.isnan(q)))

                # print("q: ", torch.any(torch.isnan(q)))
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
                # print("x: {}".format(x))
                z.append(embedding[bs * v: bs * (v + 1), :] / args.temperature)
                # print(F.log_softmax(x, dim=1))
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x+1e-3, dim=1), dim=1))
            consistency_loss += subloss / (np.sum(args.nmb_crops) - 1)
        consistency_loss /= len(args.crops_for_assign)

        contrast_loss = model.loss_cal(z[0], z[1], hard_q, neg_q, args)
        # print("contrast_loss:{}".format(contrast_loss))
        # print("sub_loss:{}".format(6 * subloss / (np.sum(args.nmb_crops) - 1)))
        if not args.reweighted_loss:
            loss = consistency_loss  #+ contrast_loss
        else:
            loss = 500 * consistency_loss  + contrast_loss

        # print("loss:{}".format(loss), "step:", step+(epoch-1)*len(loader1))
        writer.add_scalar('contrast_loss', contrast_loss, step+(epoch-1)*len(loader1))
        writer.add_scalar('consistency_loss', consistency_loss, step+(epoch-1)*len(loader1))

        writer.add_scalar('batch_loss', loss, step+(epoch-1)*len(loader1))
        writer.add_scalar('lr', lr_schedule[iteration], step+(epoch-1)*len(loader1))

        loss.backward()
        clip_grad_value_(model.parameters(), clip_value=1.1)

        # cancel gradients for the prototypes
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None

        optimizer.step()
        # for name, params in model.named_parameters():
        #     print('name:', name, 'val', params)

        if np.isnan(loss.item()):
            raise Exception('Loss value is NaN!')

        # train_loss_accum += float(loss.detach().cpu().item())
        # acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
        # acc = torch.tensor(0)
        # train_acc_accum += float(acc.detach().cpu().item())
        # print("train_loss_accum: {}".format(train_loss_accum))
    return train_acc_accum/(step+1), train_loss_accum/(step+1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
        help='dropout ratio (default: 0)')
    parser.add_argument('--temperature', default=0.2, type=float,
        help='softmax temperature')
    parser.add_argument('--JK', type=str, default="last",
        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default = 'random')
    parser.add_argument('--aug_ratio1', type=float, default = 0.2)
    parser.add_argument('--aug2', type=str, default = 'random')
    parser.add_argument('--aug_ratio2', type=float, default = 0.2)
    ########### PGCL setting ###########
    parser.add_argument("--reweighted_loss", type=bool, default=True,
        help="whether to use reweighted_loss")
    parser.add_argument("--proto_dist", type=bool, default=False,
        help="whether to use prototype distance for reweighting")
    parser.add_argument("--sample_dist", type=bool, default=True,
        help="whether to use sample distance for reweighting")
    parser.add_argument("--hard_selection", type=bool, default=False,
        help="whether to use hard selection")
    parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
        help="list of number of crops (example: [2, 6])")
    parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
        help="crops resolutions (example: [224, 96])")
    parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
        help="argument in RandomResizedCrop (example: [0.14, 0.05])")
    parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
        help="argument in RandomResizedCrop (example: [1., 0.14])")
    parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
        help="list of crops id used for computing assignments")
    parser.add_argument("--epsilon", default=0.03, type=float,
        help="regularization parameter for Sinkhorn-Knopp algorithm")
    parser.add_argument("--sinkhorn_iterations", default=3, type=int,
        help="number of iterations in Sinkhorn-Knopp algorithm")
    parser.add_argument("--feat_dim", default=128, type=int,
        help="feature dimension")
    parser.add_argument("--nmb_prototypes", default=5, type=int,
        help="number of prototypes")
    parser.add_argument("--queue_length", type=int, default=500,
        help="length of the queue (0 for no queue)")
    parser.add_argument("--epoch_queue_starts", type=int, default=3,
        help="from this epoch, we start using a queue")
    parser.add_argument("--base_lr", default=0.001, type=float, help="base learning rate")
    parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
    parser.add_argument("--freeze_prototypes_niters", default=3000, type=int,
        help="freeze the prototypes during this many iterations from the start")
    parser.add_argument("--wd", default=1e-8, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=1, type=int, help="number of warmup epochs")
    parser.add_argument("--start_warmup", default=0, type=float,
        help="initial warmup learning rate")
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                        training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=1, type=int, help="""
                        number of processes: it is set automatically and
                        should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                        it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
        help="this argument is not used and should be ignored")
    parser.add_argument('--input_model_file', type=str, default = './models_pgcl_sr/sr_10.pth', help='filename to read the model (if there is any)')
    parser.add_argument("--dump_path", type=str, default=".",
        help="experiment dump path for checkpoints and log")

    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    #set up dataset
    dataset = MoleculeDataset_aug("dataset/" + args.dataset, dataset=args.dataset)
    print(dataset)
    #to set the lr
    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)


    #set up model
    gnn = GNN_Virtualnode(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)

    model = graphcl(gnn, args.emb_dim, args.num_layer, nmb_prototypes=args.nmb_prototypes)
    # print(model)
    # if not args.input_model_file == "":
    #     model.from_pretrained(args.input_model_file)
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), eps=1e-3, lr=args.lr, weight_decay=args.decay)
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                                                                                           math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    print(optimizer)

    #visulization
    writer = SummaryWriter('./tensorboard/log')

    for epoch in range(1, args.epochs):
        print("====epoch " + str(epoch))

        train_acc, train_loss = train(args, model, device, dataset, optimizer, lr_schedule, writer, epoch)

        print(train_acc)
        print(train_loss)

        if epoch % 10 == 0:
            torch.save(gnn.state_dict(), "./models_pgcl_sr/new_" + str(epoch) + ".pth")
    writer.close()

if __name__ == "__main__":
    main()
