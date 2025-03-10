import re
import os
import gc
import sys
import time
import glob
import random
import argparse
import importlib
from pathlib import Path
from functools import wraps
from collections import OrderedDict

import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .logger import setup_logger
from .data import datasets

wandb = None


## python utils
def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


# modify from YOLOv5
def increment_path(path, exist_ok=False, sep="-", mkdir=True):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == "" else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def save_args(args, path):
    with open(path, "w") as f:
        for k, v in sorted(dict(vars(args)).items()):
            f.write(f"{k}: {str(v)}\n")


def clean_vars(*args):
    for var in args:
        del var
    # torch.cuda.empty_cache()
    gc.collect()


def default(val, def_val):
    return def_val if val is None else val


## pytorch utils
class Normalizer:
    # input is uint8 image HWC, mean and std are float32, process batched data
    # based on albumentation Normalize implementation
    def __init__(self, mean, std, device):
        self.mean = (
            torch.as_tensor(mean, dtype=torch.float, device=device)
            .reshape(1, len(mean), 1, 1)
            .contiguous()
            * 255.0
        )
        self.std = (
            torch.as_tensor(std, dtype=torch.float, device=device)
            .reshape(1, len(std), 1, 1)
            .reciprocal()
            .contiguous()
            / 255.0
        )

    def fit(self, x: torch.Tensor) -> torch.Tensor:
        x = totensor(x).type_as(self.mean)
        return (x - self.mean) * self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return ((x / self.std + self.mean)).to(torch.uint8)


def get_parser(desc="PyTorch Training"):
    parser = argparse.ArgumentParser(description=desc)

    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--data-name", type=str, required=True)
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="path to dataset repository",
    )
    parser.add_argument("--img-sz", nargs=2, type=int, default=[224, 224])
    parser.add_argument("--n-classes", type=int)
    parser.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225])

    #########################
    #### model parameters ###
    #########################
    parser.add_argument(
        "-a", "--arch", default="resnet18", type=str, help="convnet architecture"
    )
    parser.add_argument("--finetune", default="fc", type=str)

    #########################
    #### optim parameters ###
    #########################
    parser.add_argument(
        "--epochs", default=100, type=int, help="number of total epochs to run"
    )
    parser.add_argument(
        "-b", "--batch-size", default=64, type=int, help="mini-batch size"
    )
    parser.add_argument(
        "-lr", "--base-lr", default=1e-3, type=float, help="base learning rate"
    )
    parser.add_argument("--final-lr", type=float, default=0, help="final learning rate")
    parser.add_argument(
        "--momentum", default=0.9, type=float, help="momentum of SGD solver"
    )
    parser.add_argument("--wd", default=5e-2, type=float, help="weight decay")
    parser.add_argument(
        "--warmup-epochs", default=10, type=int, help="number of warmup epochs"
    )
    parser.add_argument(
        "--start-warmup", default=0, type=float, help="initial warmup learning rate"
    )

    #########################
    #### dist parameters ###
    #########################
    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument(
        "--port", default=port, type=int, help="port for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1",
        type=str,
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="""
                        number of processes: it is set automatically and
                        should not be passed as argument""",
    )
    parser.add_argument(
        "--rank",
        default=0,
        type=int,
        help="""rank of this process:
                        it is set automatically and should not be passed as argument""",
    )

    #########################
    #### other parameters ###
    #########################
    parser.add_argument("--seed", type=int, default=3407, help="seed")
    parser.add_argument(
        "-j", "--workers", default=8, type=int, help="number of data loading workers"
    )
    parser.add_argument(
        "--save-freq", type=int, default=50, help="Save the model periodically"
    )
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument(
        "--amp", action="store_true", help="use automatic mixed precision"
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--pretrain-path", type=str, default=None)
    parser.add_argument(
        "--log-path", type=str, default="./logs", help="path to log experiment"
    )
    parser.add_argument("--logger-name", type=str, default=None)
    parser.add_argument(
        "--tensorboard", action="store_true", help="use tensorboard as log tool."
    )
    parser.add_argument("--wandb", action="store_true", help="use wandb as log tool.")
    parser.add_argument("--run-group", default=None, type=str)
    parser.add_argument("--run-name", default=None, type=str)
    parser.add_argument("--run-tag", nargs="*", default=None, type=str)
    parser.add_argument("--run-type", default="train", type=str)
    parser.add_argument("--comments", default="PyTorch training", type=str)

    ###########################
    #### project parameters ###
    ###########################
    parser.add_argument("--proj-name", type=str, default="PyTorch Training")

    return parser


def fix_random_seeds(seed=3407):
    """
    Fix random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def define_wandb_summary(wandb_summary):
    global wandb
    for k, v in wandb_summary.items():
        wandb.define_metric(k, summary=v)


def init_logger(args, wandb_summary=None):
    args.logger_name = args.proj_name if args.logger_name is None else args.logger_name
    logger = setup_logger(args.log_path, name=args.logger_name)
    logger.info("============ Initialising logger ============")
    # logger.info("\n".join([sys.executable, *sys.argv]))
    commands = " ".join([sys.executable, *sys.argv]).split("--")
    logger.info("\n--".join(commands))
    logger.info("Initialise python logger successfully!")

    if args.wandb:
        global wandb
        wandb = importlib.import_module("wandb")

        wandb.init(
            project=args.proj_name,
            group=args.run_group,
            name=args.run_name,
            tags=args.run_tag,
            notes=args.comments,
            job_type=args.run_type,
            dir=args.log_path,
            config=args,
        )
        define_wandb_summary(wandb_summary)
        logger.info("Initialise wandb logger successfully!")

    tb_writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        tb_path = str(
            increment_path(f"{args.log_path}/tb_log/exp", sep="_", mkdir=True)
        )
        tb_writer = SummaryWriter(tb_path)
        logger.info("Initialise tensorboard logger successfully!")

    return logger, tb_writer


def create_eval_stat(prefix, metrics_name, classes_name, mode):
    stat = OrderedDict()
    for m in metrics_name:
        for c in classes_name:
            stat[f"{prefix}/{m}_{c}"] = mode
    return stat


def create_meters(metrics_name, classes_name, format):
    meters = OrderedDict()
    for m in metrics_name:
        for c in classes_name:
            meters[f"{m}_{c}"] = AverageMeter(f"{m}_{c}", format)
    return meters


def update_meters(
    meters,
    metrics,
    preds,
    targets,
    classes_name,
    num_classes,
    cls_weights,
    batch_size,
    **kwargs,
):
    for k, v in meters.items():
        metric_name, class_name = k.split("_")
        if class_name == "AVG":
            metric_val = 0
            for c in classes_name[:-1]:
                metric_val += meters[f"{metric_name}_{c}"].avg
            metric_val /= len(num_classes)
        else:
            class_i = classes_name.index(class_name)
            metric_val = metrics[metric_name](
                preds[class_i],
                targets[:, class_i],
                num_classes=num_classes[class_i],
                average=None,
                **kwargs,
            )[cls_weights[class_i]].item()

        v.update(metric_val, batch_size)


def generate_stat_text(train_stat, val_stat, summary_stat, metrics_name, classes_name):
    stat_text = ""
    for m in metrics_name:
        stat_text += "------\n"
        for c in classes_name:
            name = f"{m}_{c}"
            stat_text += "{}: {:.4f}/{:.4f} | ".format(
                name, train_stat[name], summary_stat[f"train/{name}"].val()
            )
            stat_text += "{:.4f}/{:.4f}\n".format(
                val_stat[name], summary_stat[f"val/{name}"].val()
            )

    return stat_text


def log_stat(args, logger, tb_writer, epoch, train_stat=None, val_stat=None):
    # tensorboard log
    if args.tensorboard:
        if train_stat:
            for k, v in train_stat.items():
                tb_writer.add_scalar(f"train/{k}", v, epoch)
        if val_stat:
            for k, v in val_stat.items():
                tb_writer.add_scalar(f"val/{k}", v, epoch)
        logger.info(f"Tensorboard logs saved!")

    # wandb log
    if args.wandb:
        global wandb
        # wandb.log({"train": train_stat, "val": val_stat}, step=epoch)
        if train_stat:
            wandb.log({f"train/{k}": v for k, v in train_stat.items()}, epoch)
        if val_stat:
            wandb.log({f"val/{k}": v for k, v in val_stat.items()}, epoch)
        logger.info(f"Wandb logs saved!")


def close_logger(args, logger, tb_writer):
    logger.info("============ Closing logger ============")

    if args.tensorboard:
        tb_writer.flush()
        tb_writer.close()

    if args.wandb:
        import shutil

        global wandb

        shutil.copyfile(
            os.path.join(args.log_path, "log.txt"),
            os.path.join(wandb.run.dir, "output.log"),
        )
        logger.info(f"Log is copied into Wandb folder!")
        wandb.finish()

    logger.info("Close logger successfully!")


def init_distributed_mode(args):
    """
    Initialize the following variables:
        - world_size
        - rank
    """

    # args for each process
    ngpus_per_node = torch.cuda.device_count()
    args.rank = args.gpu_to_work_on
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    args.dist_url = f"{args.dist_url}:{args.port}"

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    torch.cuda.set_device(args.gpu_to_work_on)
    torch.distributed.barrier()
    return


def init_dataset(
    args,
    bacth_size=None,
    num_workers=None,
    sampler="dist",
    shuffle=None,
    pin_memory=True,
    drop_last=False,
    **kwargs,
):
    # params
    bacth_size = default(bacth_size, args.batch_size)
    num_workers = default(num_workers, args.workers)

    # build data
    dataset = datasets.__dict__[args.data_name](args, **kwargs)

    # dataset sampler
    if sampler == "dist":
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = None  #  If sampler is specified, shuffle must not be specified.
    else:
        sampler = None
        if shuffle is None:
            raise ValueError("shuffle must be specified if sampler is None")

    # data normaliser
    data_normalizer = Normalizer(
        mean=args.mean, std=args.std, device=f"cuda:{args.gpu_to_work_on}"
    )

    # data loader
    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        shuffle=shuffle,
        batch_size=bacth_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dataset, loader, data_normalizer


def restart_from_checkpoint(ckp_path, logger, device, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    logger.info(f"Re-starting from checkpoint: '{ckp_path}' ...")

    if not os.path.isfile(ckp_path):
        logger.warning(f"cannot found checkpoint at '{ckp_path}'")
        exit(1)

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location=device)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                logger.info(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info(f"loaded {key} from checkpoint ...")
        else:
            logger.warning(f"failed to load {key} from checkpoint ...")

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]
                logger.info(f"loaded {var_name} from checkpoint ...")


def load_ssl_pretrain(model, weight_path, logger, module_name="encoder"):
    logger.info(f"Loading pre-trained weights from '{weight_path}' ...")

    if os.path.isfile(weight_path):
        checkpoint = torch.load(weight_path, map_location="cpu")

        # rename pre-trained keys
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith("module."):
                module_name = f"module.{module_name}"
            # retain only encoder up to before the embedding layer
            if k.startswith(module_name) and not k.startswith(f"{module_name}.fc"):
                # remove prefix
                state_dict[k[len(f"{module_name}.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        logger.info(f"loaded pre-trained model weights from '{weight_path}'")
    else:
        logger.warning(f"no model weights found at '{weight_path}'")
        logger.warning("randomly initialize model")


def totensor(x):
    if len(x.shape) == 4:  # NHWC
        return torch.as_tensor(x).permute(0, 3, 1, 2).contiguous()
    else:  # HWC
        return torch.as_tensor(x).permute(2, 0, 1).contiguous()


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class BestRecorder:
    def __init__(self, mode, best=None):
        self.mode = mode

        if best is None:
            if self.mode == "min":
                self.best = sys.maxsize
            elif self.mode == "max":
                self.best = -sys.maxsize
            else:
                print("invalid mode!")
        else:
            self.best = best

    def update(self, val):
        if self.mode == "min":
            res = val < self.best
            self.best = min(val, self.best)
            return (self.best, res)

        elif self.mode == "max":
            res = val > self.best
            self.best = max(val, self.best)
            return (self.best, res)

    def val(self):
        return self.best


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return str("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# copy from https://stackoverflow.com/a/71433508/9550209
def all_gather(q, ws, device):
    """
    Gathers tensor arrays of different lengths across multiple gpus
    
    Parameters
    ----------
        q : tensor array
        ws : world size
        device : current gpu device
        
    Returns
    -------
        all_q : list of gathered tensor arrays from all the gpus

    """
    local_size = torch.tensor(q.size(0), device=device)
    all_sizes = [torch.empty_like(local_size) for _ in range(ws)]
    dist.all_gather(all_sizes, local_size)
    max_size = torch.stack(all_sizes).max()

    size_diff = max_size.item() - local_size.item()
    if size_diff:
        padding = torch.zeros(size_diff, device=device, dtype=q.dtype)
        q = torch.cat((q, padding))

    all_qs_padded = [torch.empty_like(q) for _ in range(ws)]
    dist.all_gather(all_qs_padded, q)
    all_qs = []
    for q, size in zip(all_qs_padded, all_sizes):
        all_qs.append(q[:size])
    return all_qs