"""
evaluate SimCLR skin model
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.split(SCRIPT_DIR)[0]
ROOT_PATH = os.path.split(ROOT_PATH)[0]
sys.path.insert(0, ROOT_PATH)

import time
import shutil
import logging
import traceback
from collections import OrderedDict

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import transforms as T

from torchmetrics.functional.classification import (
    multiclass_auroc,
    multiclass_recall,
    multiclass_specificity,
    multiclass_precision,
)

from src.models.baseline import Baseline
from src.utils.misc import (
    get_parser,
    increment_path,
    fix_random_seeds,
    save_args,
    init_logger,
    init_distributed_mode,
    init_dataset,
    create_eval_stat,
    update_meters,
    log_stat,
    generate_stat_text,
    close_logger,
    clean_vars,
    create_meters,
    set_requires_grad,
    BestRecorder,
    AverageMeter,
    ProgressMeter,
)

METRICS_NAME = ["AUC", "Recall", "Spec", "Prec"]
CLASSES_NAME = ["DIAG", "PN", "BWV", "VS", "PIG", "STR", "DaG", "RS", "AVG"]
NUM_CLASSES = [5, 3, 2, 3, 3, 3, 3, 2]
CLS_WEIGHTS = [2, 2, 1, 2, 2, 2, 2, 1]


def train(
    args, loader, evaluator, criterion, optimizer, epoch, scaler, metrics
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    metric_meters = create_meters(METRICS_NAME, CLASSES_NAME, ":.4f")
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],
        prefix=f"Train epoch: [{epoch}]",
    )
    logger = logging.getLogger(args.logger_name)

    if args.finetune == "fc":
        evaluator.eval()
    else:
        evaluator.train()

    end = time.time()
    all_preds = []
    all_targets = []
    all_batch_size = 0
    for it, (derm_imgs, clinic_imgs, labels) in enumerate(loader):
        bs = derm_imgs.size(0)

        # move data to cuda
        derm_imgs = derm_imgs.cuda(non_blocking=True)
        clinic_imgs = clinic_imgs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True).long()

        # measure data loading time
        data_time.update(time.time() - end)

        # ============ forward passes ... ============
        with autocast(enabled=args.amp):
            outputs = evaluator([derm_imgs, clinic_imgs])
            loss = 0.0
            for i in range(args.num_labels):
                loss += args.label_weights[i] * criterion(outputs[i], labels[:, i])
            loss = loss / args.num_labels

        # ============ backward and optim step ... ============
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ============ misc ... ============
        losses.update(loss.item(), bs)
        all_preds.append(outputs)
        all_targets.append(labels)
        all_batch_size += bs
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank == 0 and it % args.print_freq == 0:
            logger.info(progress.display(it))

        clean_vars(derm_imgs, clinic_imgs, labels, outputs, loss)

    all_preds_ = []
    for i in range(args.num_labels):
        all_preds_.append(torch.concat([pred_j[i] for pred_j in all_preds], dim=0))
    all_targets = torch.concat(all_targets, dim=0)
    update_meters(
        metric_meters,
        metrics,
        all_preds_,
        all_targets,
        CLASSES_NAME,
        NUM_CLASSES,
        CLS_WEIGHTS,
        all_batch_size,
    )

    stat_dict = {k: v.avg for k, v in metric_meters.items()}
    stat_dict.update({"loss": losses.avg})
    return stat_dict


@torch.no_grad()
def validate(args, loader, evaluator, criterion, epoch, metrics):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    metric_meters = create_meters(METRICS_NAME, CLASSES_NAME, ":.4f")
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],
        prefix=f"Val epoch: [{epoch}]",
    )
    logger = logging.getLogger(args.logger_name)

    evaluator.eval()

    end = time.time()
    all_preds = []
    all_targets = []
    all_batch_size = 0
    for it, (derm_imgs, clinic_imgs, labels) in enumerate(loader):
        bs = derm_imgs.size(0)

        # move data to cuda
        derm_imgs = derm_imgs.cuda(non_blocking=True)
        clinic_imgs = clinic_imgs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True).long()

        # measure data loading time
        data_time.update(time.time() - end)

        # ============ forward passes ... ============
        with autocast(enabled=args.amp):
            outputs = evaluator([derm_imgs, clinic_imgs])
            loss = 0.0
            for i in range(args.num_labels):
                loss += args.label_weights[i] * criterion(outputs[i], labels[:, i])
            loss = loss / args.num_labels

        # ============ misc ... ============
        losses.update(loss.item(), bs)
        all_preds.append(outputs)
        all_targets.append(labels)
        all_batch_size += bs
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank == 0 and it % args.print_freq == 0:
            logger.info(progress.display(it))

        clean_vars(derm_imgs, clinic_imgs, labels, outputs, loss)

    all_preds_ = []
    for i in range(args.num_labels):
        all_preds_.append(torch.concat([pred_j[i] for pred_j in all_preds], dim=0))
    all_targets = torch.concat(all_targets, dim=0)
    update_meters(
        metric_meters,
        metrics,
        all_preds_,
        all_targets,
        CLASSES_NAME,
        NUM_CLASSES,
        CLS_WEIGHTS,
        all_batch_size,
    )

    stat_dict = {k: v.avg for k, v in metric_meters.items()}
    stat_dict.update({"loss": losses.avg})
    return stat_dict


def main(local_rank, args):
    logger = logging.getLogger()
    train_stat = create_eval_stat("train", METRICS_NAME, CLASSES_NAME, "max")
    val_stat = create_eval_stat("val", METRICS_NAME, CLASSES_NAME, "max")
    summary_stat = OrderedDict(train_stat, **val_stat)
    summary_stat.update({"train/loss": "min", "val/loss": "min"})

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

    train_trans = T.Compose(
        [
            T.RandomResizedCrop((args.img_sz[0], args.img_sz[1]), scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(args.mean, args.std),
        ]
    )
    val_trans = T.Compose(
        [
            T.Resize((args.img_sz[0], args.img_sz[1])),
            T.ToTensor(),
            T.Normalize(args.mean, args.std),
        ]
    )

    # build train dataset
    train_dataset, train_loader, train_data_norm = init_dataset(
        args, data_trans=train_trans, mode="train"
    )
    del train_data_norm
    logger.info(
        "Building train data done with {} images loaded.".format(len(train_dataset))
    )

    # build val dataset
    val_dataset, val_loader, val_data_norm = init_dataset(
        args, sampler=None, shuffle=False, data_trans=val_trans, mode="test"
    )
    del val_data_norm
    logger.info(
        "Building val data done with {} images loaded.".format(len(val_dataset))
    )

    # init mixed precision
    if args.amp:
        logger.info(f"Enable automatic mix precision training!")
    scaler = GradScaler(enabled=args.amp)

    # build model
    logger.info(f"Building model '{args.arch}'")
    evaluator = Baseline(args.arch, args.arch_weights)
    
    # todo: harcoded
    if os.path.isfile(args.pretrain_path):
        logger.info(f"Loading pre-trained weights from '{args.pretrain_path}' ...")
        state_dict = torch.load(args.pretrain_path, map_location="cpu")["state_dict"]
        derm_state_dict = {}
        clinic_state_dict = {}

        for k in state_dict.keys():
            k_strim = k
            if k_strim.startswith("module."):
                k_strim = k_strim[7:]
                
            if k_strim.startswith("derm_backbone.encoder."):
                derm_state_dict[k_strim[len("derm_backbone.encoder.") :]] = state_dict[k]
            elif k_strim.startswith("clinic_backbone.encoder."):
                clinic_state_dict[k_strim[len("clinic_backbone.encoder.") :]] = state_dict[k]

        evaluator.derm_backbone.load_state_dict(derm_state_dict)
        evaluator.clinic_backbone.load_state_dict(clinic_state_dict)
        logger.info(f"loaded pre-trained model weights from '{args.pretrain_path}'")
    
    else:
        logger.warning(f"Pre-trained weights not found at '{args.pretrain_path}'")
        logger.info(f"Initialised model weights with '{args.arch_weights}' ...")

    if args.finetune == "fc":
        set_requires_grad(evaluator.derm_backbone, False)
        set_requires_grad(evaluator.clinic_backbone, False)

    for name, params in evaluator.named_parameters():
        logger.info(f"{name}: {params.requires_grad}")

    # todo: need test
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # copy model to GPU
    evaluator = evaluator.cuda()
    if args.rank == 0:
        logger.info(evaluator)
    logger.info("Building model done.")

    # wrap model
    evaluator = DDP(evaluator, device_ids=[args.gpu_to_work_on])

    # build optimizer
    parameters = list(filter(lambda p: p.requires_grad, evaluator.parameters()))
    optimizer = torch.optim.AdamW(
        parameters, lr=args.base_lr, weight_decay=args.wd, eps=1e-5
    )
    logger.info("Building optimizer done.")

    # build loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # initialize metric
    train_metrics = {
        METRICS_NAME[0]: multiclass_auroc,
        METRICS_NAME[1]: multiclass_recall,
        METRICS_NAME[2]: multiclass_specificity,
        METRICS_NAME[3]: multiclass_precision,
    }
    val_metrics = train_metrics

    start_epoch = 0
    best_auc = 0
    cudnn.benchmark = True

    for k, v in summary_stat.items():
        summary_stat[k] = BestRecorder(v)
    summary_stat["val/AUC_AVG"].best = best_auc

    for epoch in range(start_epoch, args.epochs):
        end = time.time()

        # train the network for one epoch
        logger.info(f"============ Starting epoch {epoch} ... ============")

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        train_stat = train(
            args,
            train_loader,
            evaluator,
            criterion,
            optimizer,
            epoch,
            scaler,
            train_metrics,
        )
        summary_stat["train/loss"].update(train_stat["loss"])
        for m in METRICS_NAME:
            for c in CLASSES_NAME:
                summary_stat[f"train/{m}_{c}"].update(train_stat[f"{m}_{c}"])

        # eval step
        val_stat = validate(
            args,
            val_loader,
            evaluator,
            criterion,
            epoch,
            val_metrics,
        )
        summary_stat["val/loss"].update(val_stat["loss"])
        for m in METRICS_NAME:
            for c in CLASSES_NAME:
                summary_stat[f"val/{m}_{c}"].update(val_stat[f"{m}_{c}"])
        is_best = summary_stat["val/AUC_AVG"].val() > best_auc

        # save and log
        if args.rank == 0:
            # save model
            logger.info("Saving model ...")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": evaluator.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_auc": summary_stat["val/AUC_AVG"].val(),
            }
            torch.save(
                save_dict,
                os.path.join(args.log_path, "checkpoint.pth.tar"),
            )
            
            # save best
            if is_best:
                logger.info("Saving best model ...")
                shutil.copyfile(
                    os.path.join(args.log_path, "checkpoint.pth.tar"),
                    os.path.join(args.log_path, "best_eval.pth"),
                )
                best_auc = summary_stat["val/AUC_AVG"].val()

            # log stat
            log_stat(args, logger, tb_writer, epoch, train_stat, val_stat)
            stat_text = generate_stat_text(
                train_stat, val_stat, summary_stat, METRICS_NAME, CLASSES_NAME
            )

            # epoch summary log
            epoch_time = (time.time() - end) / 60
            logger.info(
                f"--------------------- Summary Statistics -----------------------\n"
                f"Elapse Time: {epoch_time:.2f} mins\n"
                f"Loss: {train_stat['loss']:.4f} | {val_stat['loss']:.4f}\n"
                f"{stat_text}"
            )

    # run linear probing
    if args.rank == 0:
        close_logger(args, logger, tb_writer)


if __name__ == "__main__":
    parser = get_parser("PyTorch SimCLR Skin Finetuning")
    parser.add_argument("--arch-weights", type=str, default=None)
    parser.add_argument("--num-labels", type=int, default=8)
    parser.add_argument(
        "--label-weights", type=float, default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
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
