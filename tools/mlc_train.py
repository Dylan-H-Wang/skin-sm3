"""
copy from train_v3.py
change SA to transformer encoder layer
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.split(SCRIPT_DIR)[0]
sys.path.insert(0, ROOT_PATH)

import time
import shutil
import logging
import traceback

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import transforms as T

import numpy as np
from scipy.sparse import csr_matrix

from src.models.simclr import SimCLRSkinV32
from src.models.projector import *
from src.utils.misc import (
    get_parser,
    increment_path,
    fix_random_seeds,
    save_args,
    init_logger,
    init_distributed_mode,
    init_dataset,
    log_stat,
    close_logger,
    clean_vars,
    set_requires_grad,
    BestRecorder,
    AverageMeter,
    ProgressMeter,
)

METRICS_NAME = ["AUC", "Recall", "Spec", "Prec"]
CLASSES_NAME = ["DIAG", "PN", "BWV", "VS", "PIG", "STR", "DaG", "RS", "AVG"]
NUM_CLASSES = [5, 3, 2, 3, 3, 3, 3, 2]
CLS_WEIGHTS = [2, 2, 1, 2, 2, 2, 2, 1]
        

class Model(nn.Module):
    def __init__(self, extractor, projectors, feat_dim, l2_norm, n_heads, sa_dim_ff, sa_dropout):
        super().__init__()
        self.extractor = extractor
        self.projectors = projectors
        self.mlc_sa = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=n_heads, dim_feedforward=sa_dim_ff, dropout=sa_dropout)
        # DeepCluster do not need bias
        self.prototypes = nn.ModuleList(
            [nn.Linear(feat_dim, i, bias=False) for i in NUM_CLASSES]
        )
        self.l2_norm = l2_norm

    def forward(self, derm_imgs, clinic_imgs):
        feats = self.extractor.extract(derm_imgs, clinic_imgs)
        feats = torch.cat(feats, dim=1)

        proj_feats = self.projectors(feats)
        if not isinstance(proj_feats, list):
            proj_feats = [proj_feats]

        proj_feats = torch.stack(proj_feats, dim=0)
        sa_feats = self.mlc_sa(proj_feats)
            
        if self.l2_norm:
            for i in range(len(sa_feats)):
                sa_feats[i] = nn.functional.normalize(sa_feats[i], dim=-1, p=2)

        preds = [
            self.prototypes[i](sa_feats[i%len(sa_feats)]) for i in range(len(self.prototypes))
        ]

        return sa_feats, preds


@torch.no_grad()
def init_memory(logger, dataloader, model):
    local_memory_index = []
    local_memory_embeddings = []

    logger.info("Start initializing the memory banks")
    for index, inputs in dataloader:
        index = index.cuda(non_blocking=True)

        # get embeddings
        derm_imgs = inputs[0].cuda(non_blocking=True)
        clinic_imgs = inputs[1].cuda(non_blocking=True)
        outputs, _ = model(derm_imgs, clinic_imgs)

        # fill the memory bank
        local_memory_index.append(index)
        local_memory_embeddings.append(outputs)
    local_memory_index = torch.cat(local_memory_index, dim=0).cuda().long()
    local_memory_embeddings = torch.cat(local_memory_embeddings, dim=1).cuda()
    logger.info("Initializion of the memory banks done.")

    return local_memory_index, local_memory_embeddings


@torch.no_grad()
def cluster_memory(
    args,
    prototypes,
    K,
    local_memory_index,
    local_memory_embeddings,
    nmb_kmeans_iters=10,
):
    assignments = -100 * torch.ones(
        len(local_memory_index) * args.world_size, dtype=torch.long
    ).cuda(non_blocking=True)
    centroids = torch.empty(K, local_memory_embeddings.size(1)).cuda(non_blocking=True)
    all_local_memory_embeddings = [
        torch.empty_like(local_memory_embeddings) for _ in range(args.world_size)
    ]
    all_local_memory_index = [
        torch.empty_like(local_memory_index) for _ in range(args.world_size)
    ]

    if args.rank == 0:
        dist.gather(local_memory_embeddings, all_local_memory_embeddings)
        dist.gather(local_memory_index, all_local_memory_index)
        all_local_memory_embeddings = torch.cat(all_local_memory_embeddings, dim=0)
        all_local_memory_index = torch.cat(all_local_memory_index, dim=0)
    else:
        dist.gather(local_memory_embeddings)
        dist.gather(local_memory_index)

    if args.rank == 0:
        random_idx = torch.randperm(len(all_local_memory_embeddings))[:K]
        centroids = all_local_memory_embeddings[random_idx]
        assert len(random_idx) >= K, "please reduce the number of centroids"

        for n_iter in range(nmb_kmeans_iters + 1):

            # E step
            dot_products = torch.mm(all_local_memory_embeddings, centroids.t())
            _, local_assignments = dot_products.max(dim=1)

            # finish
            if n_iter == nmb_kmeans_iters:
                break

            # M step
            where_helper = get_indices_sparse(local_assignments.cpu().numpy())
            counts = torch.zeros(K).cuda(non_blocking=True).int()
            emb_sums = torch.zeros(K, local_memory_embeddings.size(1)).cuda(
                non_blocking=True
            )
            for k in range(len(where_helper)):
                if len(where_helper[k][0]) > 0:
                    emb_sums[k] = torch.sum(
                        all_local_memory_embeddings[where_helper[k][0]],
                        dim=0,
                    )
                    counts[k] = len(where_helper[k][0])
            mask = counts > 0
            centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

            # normalize centroids
            centroids = nn.functional.normalize(centroids, dim=1, p=2)

        assignments_all = local_assignments
        indexes_all = all_local_memory_index

        # log assignments
        assignments[indexes_all] = assignments_all

    dist.broadcast(centroids, 0)
    dist.broadcast(assignments, 0)
    prototypes.weight.copy_(centroids)

    return assignments


def get_indices_sparse(data):
    cols = np.arange(data.size)
    M = csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
    return [np.unravel_index(row.data, data.shape) for row in M]


def ssl_train(
    args,
    loader,
    model,
    criterion,
    optimizer,
    epoch,
    local_memory_index,
    local_memory_embeddings,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],
        prefix=f"Train epoch: [{epoch}]",
    )
    logger = logging.getLogger(args.logger_name)

    all_assignments = []
    for idx, prototype in enumerate(model.module.prototypes):
        all_assignments.append(
            cluster_memory(
                args,
                prototype,
                prototype.weight.size(0),
                local_memory_index,
                local_memory_embeddings[idx%len(local_memory_embeddings)],
            )
        )
    logger.info("Clustering for epoch {} done.".format(epoch))

    if args.finetune_backbone:
        model.train()
    else:
        model.module.extractor.eval()
        model.module.projectors.train()
        model.module.mlc_sa.train()
        model.module.prototypes.train()

    start_idx = 0
    end = time.time()
    for it, (idx, (derm_imgs, clinic_imgs, _)) in enumerate(loader):
        bs = derm_imgs.size(0)

        # move data to cuda
        derm_imgs = derm_imgs.cuda(non_blocking=True)
        clinic_imgs = clinic_imgs.cuda(non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        # ============ forward passes ... ============
        proj_feats, preds = model(derm_imgs, clinic_imgs)

        # ============ calculate loss ... ============
        loss = 0
        for pred, assignment in zip(preds, all_assignments):
            scores = pred / args.temperature
            targets = assignment[idx].cuda(non_blocking=True)
            loss += criterion(scores, targets)
            # todo: add intra-cluster constrain
        loss = loss / len(all_assignments)

        # ============ backward and optim step ... ============
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # ============ update memory banks ... ============
        local_memory_index[start_idx : start_idx + bs] = idx
        for i in range(len(local_memory_embeddings)):
            local_memory_embeddings[i][start_idx : start_idx + bs] = proj_feats[
                i
            ].detach()
        start_idx += bs

        # ============ misc ... ============
        losses.update(loss.item(), bs)
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank == 0 and it % args.print_freq == 0:
            logger.info(progress.display(it))

        clean_vars(derm_imgs, clinic_imgs, proj_feats, preds, loss)

    return {"loss": losses.avg}


def main(local_rank, args):
    logger = logging.getLogger()
    summary_stat = {
        "ssl_train/loss": "min",
    }

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # only log if master(rank0)
    if local_rank == 0:
        logger, tb_writer = init_logger(args, summary_stat)

    args.gpu_to_work_on = local_rank
    fix_random_seeds(args.seed)
    init_distributed_mode(args)
    logger.info(
        "============ Initialising distributed mode ============\n"
        f"rank: {args.rank}\nbatch size: {args.batch_size}\nnum_workers: {args.workers}"
    )

    train_data_trans = T.Compose(
        [
            T.RandomResizedCrop((args.img_sz[0], args.img_sz[1]), scale=(0.5, 1.0)),
            T.RandomApply(
                [
                    T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                ],
                p=0.5,
            ),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                mean=args.mean,
                std=args.std,
            ),
        ]
    )

    # build dataset
    train_dataset, train_loader, train_data_norm = init_dataset(
        args, data_trans=train_data_trans, mode="train", return_index=True
    )
    del train_data_norm
    logger.info(
        "Building train data done with {} images loaded.".format(len(train_dataset))
    )

    # build model
    extractor = SimCLRSkinV32(
        arch=args.arch,
        proj_dim=args.extractor_proj_dim,
    )
    extractor.load_state_dict(
        torch.load(args.extractor_weights, map_location="cpu")["state_dict"]
    )
    extractor.derm_backbone.projector = None
    extractor.clinic_backbone.projector = None
    extractor.cross_proj = None
    if not args.finetune_backbone:
        set_requires_grad(extractor, False)
    
    feat_dim = extractor.derm_feat_dim + extractor.clinic_feat_dim
    
    if args.mlc_proj == "v0":
        projectors = nn.Identity()
    elif args.mlc_proj == "v1":
        projectors = MultiLabelProjector(feat_dim, args.mlc_proj_dim, args.num_labels)
    elif args.mlc_proj == "v2":
        projectors = MultiLabelProjector2(feat_dim, args.mlc_proj_dim, args.num_labels)
    elif args.mlc_proj == "v3":
        projectors = MultiLabelProjector3(feat_dim, args.mlc_proj_dim, args.num_labels)
    elif args.mlc_proj == "v4":
        projectors = MultiLabelProjector4(feat_dim, args.mlc_proj_dim, args.num_labels)

    model = Model(extractor, projectors, args.mlc_proj_dim, args.l2_norm, args.num_heads, args.sa_dim_ff, args.sa_dropout)
    model = model.cuda()

    if args.rank == 0:
        logger.info(model)
        for name, params in model.named_parameters():
            logger.info(f"{name}: {params.requires_grad}")
    logger.info("Building model done.")

    # wrap model
    model = DDP(model, device_ids=[args.gpu_to_work_on])

    # build optimizer
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(parameters, lr=args.base_lr, weight_decay=args.wd)
    logger.info("Building optimizer done.")

    # build loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-100).cuda()

    # build the memory bank
    local_memory_index, local_memory_embeddings = init_memory(
        logger, train_loader, model
    )

    cudnn.benchmark = True

    for k, v in summary_stat.items():
        summary_stat[k] = BestRecorder(v)

    for epoch in range(args.epochs):
        end = time.time()

        # train the network for one epoch
        logger.info(f"============ Starting epoch {epoch} ... ============")

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        train_stat = ssl_train(
            args,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            local_memory_index,
            local_memory_embeddings,
        )
        summary_stat["ssl_train/loss"].update(train_stat["loss"])

        # save and log
        if args.rank == 0:
            # save model
            if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
                logger.info("Saving model ...")
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": model.module.state_dict(),
                }
                torch.save(
                    save_dict,
                    os.path.join(args.log_path, f"ckp_{epoch}.pth"),
                )

            # log stat
            log_stat(args, logger, tb_writer, epoch, train_stat)

            # epoch summary log
            epoch_time = (time.time() - end) / 60
            logger.info(
                f"Elapse Time: {epoch_time:.2f} mins, Train Loss: {train_stat['loss']:.4f}/{summary_stat['ssl_train/loss'].val():.4f}"
            )

    clean_vars(train_dataset, train_loader, optimizer)

    # run linear probing
    if args.rank == 0:
        close_logger(args, logger, tb_writer)


if __name__ == "__main__":
    parser = get_parser("SM3 DeepCluster Training v3")
    parser.add_argument("--num-labels", type=int, default=8)
    parser.add_argument("--extractor-proj-dim", type=int, default=128)
    parser.add_argument("--extractor-weights", type=str, default=None)
    parser.add_argument("--mlc-proj", type=str)
    parser.add_argument("--mlc-proj-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--sa-dim-ff", type=int, default=256)
    parser.add_argument("--sa-dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1) 
    parser.add_argument("--l2-norm", action="store_true")
    parser.add_argument("--finetune-backbone", action="store_true")
    args = parser.parse_args()

    args.world_size = torch.cuda.device_count()
    args.log_path = str(increment_path(args.log_path, sep="_", mkdir=True))
    save_args(args, os.path.join(args.log_path, "configs.txt"))

    try:
        mp.spawn(main, nprocs=args.world_size, args=(args,))

    except Exception as e:
        print(e, "\n")
        # print origin trace info
        with open(args.log_path + "/error.log", "a") as myfile:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info, file=myfile)
            myfile.write("\n")
            del exc_info
