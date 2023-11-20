import sys 
from distributed import is_master, init_distributed_device, broadcast_object


### https://github.com/mlfoundations/open_clip/blob/main/src/training.logger.py
import logging


def setup_logging(log_file, level, include_host=False):
    if include_host:
        import socket
        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f'%(asctime)s |  {hostname} | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    else:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


### https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False


def gather_features(set_of_features, rank=0, world_size=1):

    def gather_features_one(features):
        gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
        dist.all_gather(gathered_features, features)

        # ensure grads for local rank when all_* features don't have a gradient
        gathered_features[rank] = features

        all_features = torch.cat(gathered_features, dim=0)
        return all_features

    all_features = [gather_features_one(features) for features in set_of_features]

    return all_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                [image_features, text_features],
                rank=self.rank, world_size=self.world_size)

            logits_per_image = logit_scale * all_image_features @ all_text_features.T
            logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        return total_loss



import argparse
def get_default_params(args):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = f'model-bs{args.batch_size}'
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers per GPU.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args


class TestNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.net = nn.Linear(size, size)
        with torch.no_grad():
            self.net.weight.fill_(1.)
            self.net.bias.fill_(0.)
    def forward(self, x):
        return self.net(x)


class TestDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return torch.rand(768), torch.rand(768)
    def __len__(self):
        return 10000


def main(args):
    args = parse_args(args)

    setup_logging('./log.txt', logging.DEBUG)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    args.device = init_distributed_device(args)

    if args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    # Test instances
    model_x = TestNetwork(768).to(args.device)
    model_y = TestNetwork(768).to(args.device)
    ds = TestDataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size)
    clip_loss = ClipLoss(rank=args.rank, world_size=args.world_size)

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model_x)}")
        logging.info(f"{str(model_y)}")

    if args.distributed:
        model_x = torch.nn.parallel.DistributedDataParallel(model_x, device_ids=[args.device])
        model_y = torch.nn.parallel.DistributedDataParallel(model_y, device_ids=[args.device])

    for epoch in range(args.epochs):
        for x, y in dl:
            x, y = x.to(args.device), y.to(args.device)
            logging.info(f'rank={args.rank}: x.shape={x.shape}, x[:2, 0]={x[:2, 0]}, y.shape={y.shape}, y[:2, 0]={y[:2, 0]}')

            x = model_x(x)
            y = model_y(y)

            all_x, all_y = gather_features([x, y], world_size=args.world_size)
            logging.info(f'rank={args.rank}: all_x.shape={all_x.shape}, [:2, 0]={all_x[:2, 0]}, all_y.shape={all_y.shape}, [:2, 0]={all_y[:2, 0]}')

            loss = clip_loss(all_x, all_y, logit_scale=1.)
            logging.info(f'rank={args.rank}, clip loss={loss}')
            break
        break


if __name__ == "__main__":
    main(sys.argv[1:])