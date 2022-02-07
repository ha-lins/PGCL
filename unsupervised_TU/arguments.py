import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const',
        const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const',
        const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const',
        const=True, default=False)
    parser.add_argument('--dropout_rate', dest='dropout_rate', type=float, default=0.1,
                        help='Dropout rate.')
    parser.add_argument('--lr', dest='lr', type=float, default=0.01,
        help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
        help='Number of graph convolution layers before each pooling')

    parser.add_argument('--bs', dest='bs', type=int, default=32,
                        help='batch_size')

    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
        help='')

    parser.add_argument('--aug', type=str, default='subgraph')
    parser.add_argument('--stro_aug', type=str, default='stro_dnodes')
    parser.add_argument('--weak_aug2', type=str, default=None)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_momentum', type=bool, default=True)

    #OGB ARGS
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')

    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of workers (default: 0)')
    # PCL args
    parser.add_argument('--num-cluster', default=[6], type=int,
                        help='number of clusters')
    parser.add_argument('--warmup-epoch', default=5, type=int,
                        help='number of warm-up epochs to only train with InfoNCE loss')
    parser.add_argument('--exp-dir', default='experiment_pcl', type=str,
                        help='experiment directory')
    parser.add_argument('--low-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--temperature', default=0.2, type=float,
                        help='softmax temperature')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')

    # SwaV args
    parser.add_argument("--vis_flag", type=bool, default=True,
                        help="whether to visualize")
    parser.add_argument("--sample_reweighting", type=bool, default=False,
                        help="whether to visualize")
    parser.add_argument("--hard_selection", type=bool, default=True,
                        help="whether to visualize")
    parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                        help="path to dataset repository")
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
    parser.add_argument("--epsilon", default=0.05, type=float,
                        help="regularization parameter for Sinkhorn-Knopp algorithm")
    parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                        help="number of iterations in Sinkhorn-Knopp algorithm")
    parser.add_argument("--feat_dim", default=128, type=int,
                        help="feature dimension")
    parser.add_argument("--nmb_prototypes", default=10, type=int,
                        help="number of prototypes")
    parser.add_argument("--queue_length", type=int, default=1800,
                        help="length of the queue (0 for no queue)")
    parser.add_argument("--epoch_queue_starts", type=int, default=3,
                        help="from this epoch, we start using a queue")
    parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
    parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
    parser.add_argument("--freeze_prototypes_niters", default=3, type=int,
                        help="freeze the prototypes during this many iterations from the start")
    parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
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
    parser.add_argument("--dump_path", type=str, default=".",
                        help="experiment dump path for checkpoints and log")

    return parser.parse_args()

