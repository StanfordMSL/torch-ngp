import argparse
import os
import glob
import sys
import tqdm
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from kornia import create_meshgrid

from scipy.spatial.transform import Rotation
import lietorch


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5

    i, j = grid.unbind(-1)

    if center is None:
        center = [W / 2, H / 2]

    directions = torch.stack(
        [(i - center[0]) / focal[0], (j - center[1]) / focal[1], torch.ones_like(i)],
        dim=-1,
    )  # (H, W, 3)

    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = lietorch.SO3(c2w).act(directions)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w.translation()[..., :3].expand(rays_d.shape)  # (H, W, 3)

    return rays_o, rays_d


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(
                        xs, ys, zs, indexing="ij"
                    )  # for torch < 1.10, should remove indexing='ij'
                    pts = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                        dim=-1,
                    ).unsqueeze(
                        0
                    )  # [1, N, 3]
                    val = (
                        query_func(pts)
                        .reshape(len(xs), len(ys), len(zs))
                        .detach()
                        .cpu()
                        .numpy()
                    )  # [1, N, 1] --> [x, y, z]
                    u[
                        xi * N : xi * N + len(xs),
                        yi * N : yi * N + len(ys),
                        zi * N : zi * N + len(zs),
                    ] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    # print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    # print(u.shape, u.max(), u.min(), np.percentile(u, 50))

    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = (
        vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :]
        + b_min_np[None, :]
    )
    return vertices, triangles


def load_model_checkpoint(
    model, ckpt_path, name, checkpoint=None, device=None, log=None
):

    if log is None:
        log = lambda x: None

    if checkpoint is None:
        checkpoint_list = sorted(glob.glob(f"{ckpt_path}/{name}_ep*.pth.tar"))
        if checkpoint_list:
            checkpoint = checkpoint_list[-1]
            log(f"[INFO] Latest checkpoint is {checkpoint}")
        else:
            log("[WARN] No checkpoint found, model randomly initialized.")
            return

    checkpoint_dict = torch.load(checkpoint, map_location=device)

    if "model" not in checkpoint_dict:
        model.load_state_dict(checkpoint_dict)
        log("[INFO] loaded model.")
        return

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint_dict["model"], strict=False
    )
    log("[INFO] loaded model.")
    if len(missing_keys) > 0:
        log(f"[WARN] missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        log(f"[WARN] unexpected keys: {unexpected_keys}")

    if model.cuda_ray:
        if "mean_count" in checkpoint_dict:
            model.mean_count = checkpoint_dict["mean_count"]
        if "mean_density" in checkpoint_dict:
            model.mean_density = checkpoint_dict["mean_density"]

    return model, checkpoint_dict


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean(np.power(preds - truths, 2)))

        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f"PSNR = {self.measure():.6f}"


class Trainer(object):
    def __init__(
        self,
        name,  # name of this experiment
        conf,  # extra conf
        model,  # network
        criterion=None,  # loss function, if None, assume inline implementation in train_step
        train_pose_vars=None,  # poses to use for train images
        optimizer=None,  # optimizer
        pose_optimizer=None,  # optimizer for camera poses
        ema_decay=None,  # if use EMA, set the decay
        lr_scheduler=None,  # scheduler
        pose_scheduler=None,
        metrics=[],  # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
        local_rank=0,  # which GPU am I
        world_size=1,  # total num of GPUs
        device=None,  # device to use, usually setting to None is OK. (auto choose device)
        mute=False,  # whether to mute all print
        fp16=False,  # amp optimize level
        eval_interval=1,  # eval once every $ epoch
        max_keep_ckpt=2,  # max num of saved ckpts in disk
        workspace="workspace",  # workspace to save logs & ckpts
        best_mode="min",  # the smaller/larger result, the better
        use_loss_as_metric=True,  # use loss as the first metric
        report_metric_at_train=False,  # also report metrics at training
        use_checkpoint="latest",  # which ckpt to use at init time
        use_tensorboardX=True,  # whether to use tensorboard for logging
        scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
    ):

        self.name = name
        self.conf = conf
        self.train_pose_vars = train_pose_vars
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = (
            device
            if device is not None
            else torch.device(
                f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            )
        )
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank]
            )
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=0.001, weight_decay=5e-4
            )  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        self.pose_optimizer = pose_optimizer

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1
            )  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        self.pose_scheduler = pose_scheduler

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=ema_decay
            )
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # a persistent iterator of dataloader, to support custom max_steps_per_epoch
        self.loader = None

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = "min"

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, "checkpoints")
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}'
        )
        self.log(
            f"[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}"
        )

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    ### ------------------------------

    def train_step(self, data):
        if self.pose_optimizer:

            train_poses = self.train_pose_vars.retr()
            train_poses.data = train_poses.data.reshape(-1, 1, 1, 7).expand(
                -1, self.H, self.W, -1
            )
            train_poses.data = train_poses.data.reshape(-1, 7)

            rays_o, rays_d = get_rays(data["directions"], train_poses[data["index"]])

            # TODO(pculbert): This is a dirty, dirty hack.
            # Don't mutate things randomly, kids.
            data["train_poses"] = train_poses[data["index"]]

        else:

            rays_o, rays_d = get_rays(
                data["directions"], lietorch.SE3(data["pose_data"])
            )

        rgbs = data["rgbs"]  # [N, 3/4]

        N, C = rgbs.shape

        # train with random background color if using alpha mixing
        # bg_color = torch.ones(3, device=self.device) # [3], fixed white background
        bg_color = torch.rand(3, device=self.device)  # [3], frame-wise random.
        if C == 4:
            rgbs = rgbs[..., :3] * rgbs[..., 3:] + bg_color * (1 - rgbs[..., 3:])

        outputs = self.model.render(
            rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, **self.conf
        )

        pred_rgb = outputs["rgb"]

        loss = self.criterion(pred_rgb, rgbs)

        #         if self.pose_optimizer:
        #             pose_loss = 1e-1 * torch.nn.MSELoss()(train_poses[data['index']].vec(),
        #                                        lietorch.SE3(data['pose_data']).vec())

        #             loss += pose_loss

        return pred_rgb, rgbs, loss

    def eval_step(self, data):
        rays_o, rays_d = get_rays(data["directions"], lietorch.SE3(data["pose_data"]))
        rgbs = data["rgbs"]  # [B, H, W, 3/4]

        B, H, W, C = rgbs.shape
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        # eval with fixed background color
        bg_color = torch.ones(3, device=self.device)  # [3]
        if C == 4:
            rgbs = rgbs[..., :3] * rgbs[..., 3:] + bg_color * (1 - rgbs[..., 3:])

        outputs = self.model.render(
            rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **self.conf
        )

        pred_rgb = outputs["rgb"].reshape(B, H, W, -1)
        pred_depth = outputs["depth"].reshape(B, H, W)

        loss = self.criterion(pred_rgb, rgbs)

        return pred_rgb, pred_depth, rgbs, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):

        # Support case for GUI test.
        if "rays_o" in data:
            rays_o, rays_d = data["rays_o"], data["rays_d"]
        else:
            rays_o, rays_d = get_rays(
                data["directions"], lietorch.SE3(data["pose_data"])
            )
        B, H, W, _ = rays_o.shape

        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(
            rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, **self.conf
        )

        pred_rgb = outputs["rgb"].reshape(B, H, W, -1)
        pred_depth = outputs["depth"].reshape(B, H, W)

        return pred_rgb, pred_depth

    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(
                self.workspace, "meshes", f"{self.name}_{self.epoch}.ply"
            )

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sdfs = self.model.density(pts.to(self.device))
            return sdfs

        bounds_min = torch.FloatTensor([-self.model.bound] * 3)
        bounds_max = torch.FloatTensor([self.model.bound] * 3)

        vertices, triangles = extract_geometry(
            bounds_min,
            bounds_max,
            resolution=resolution,
            threshold=threshold,
            query_func=query_func,
        )

        mesh = trimesh.Trimesh(
            vertices, triangles, process=False
        )  # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs, max_steps_per_epoch=-1):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(
                os.path.join(self.workspace, "run", self.name)
            )

        # to support max_steps_per_epoch
        self.loader = iter(train_loader)

        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader, max_steps_per_epoch)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(
        self,
        loader,
    ):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, "results")

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(
            total=len(loader),
            bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        self.model.eval()
        with torch.no_grad():

            if self.model.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            for i, data in enumerate(loader):

                data = self.prepare_data(data)

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)

                path = os.path.join(save_path, f"{i:04d}.png")
                path_depth = os.path.join(save_path, f"{i:04d}_depth.png")

                # self.log(f"[INFO] saving test image to {path}")

                cv2.imwrite(
                    path,
                    cv2.cvtColor(
                        (preds[0].detach().cpu().numpy() * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR,
                    ),
                )
                cv2.imwrite(
                    path_depth,
                    (preds_depth[0].detach().cpu().numpy() * 255).astype(np.uint8),
                )

                pbar.update(1)

        self.log(f"==> Finished Test.")

    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        # update grid
        if self.model.cuda_ray:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        # may cause a very slow starting...
        if self.loader is None:
            self.loader = iter(train_loader)

        for _ in range(step):

            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(self.loader)
            except StopIteration:
                self.loader = iter(train_loader)
                data = next(self.loader)

            self.global_step += 1

            data = self.prepare_data(data)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        if self.pose_scheduler:
            self.pose_scheduler.step()

        outputs = {
            "loss": average_loss,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        return outputs

    # [GUI] test on a single image
    def test_gui(self, rays_o, rays_d, bg_color=None, spp=1):

        data = {
            "rays_o": rays_o.unsqueeze(0),
            "rays_d": rays_d.unsqueeze(0),
        }

        data = self.prepare_data(data)

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                preds, preds_depth = self.test_step(
                    data, bg_color=bg_color, perturb=spp
                )

        if self.ema is not None:
            self.ema.restore()

        outputs = {
            "image": preds[0].detach().cpu().numpy(),
            "depth": preds_depth[0].detach().cpu().numpy(),
        }

        return outputs

    def prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, non_blocking=True)
        else:  # is_tensor, or other similar objects that has `to`
            data = data.to(self.device, non_blocking=True)

        return data

    def train_one_epoch(self, train_loader, max_steps_per_epoch=-1):
        self.log(
            f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ..."
        )

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # update grid
        if self.model.cuda_ray:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            train_loader.sampler.set_epoch(self.epoch)

        if max_steps_per_epoch == -1:
            max_steps_per_epoch = len(train_loader)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=max_steps_per_epoch,
                bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        self.local_step = 0

        for _ in range(max_steps_per_epoch):

            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(self.loader)
            except StopIteration:
                self.loader = iter(train_loader)
                data = next(self.loader)

            if self.local_step >= max_steps_per_epoch:
                break

            self.local_step += 1
            self.global_step += 1

            data = self.prepare_data(data)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)

            # Memory hack -- compute the pose reg grads separately.
            if self.pose_optimizer:
                pose_loss = 1e-3 * torch.nn.MSELoss()(
                    data["train_poses"].vec(), lietorch.SE3(data["pose_data"]).vec()
                )

                # This adds the regularization grads to the existing loss grads
                # for the pose vars.
                self.scaler.scale(pose_loss).backward(retain_graph=True)

            self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)

            if self.pose_optimizer:
                self.scaler.step(self.pose_optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar(
                        "train/lr",
                        self.optimizer.param_groups[0]["lr"],
                        self.global_step,
                    )

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                else:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                    )
                pbar.update(1)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader),
                bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        with torch.no_grad():
            self.local_step = 0

            if self.model.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            for data in loader:
                self.local_step += 1

                data = self.prepare_data(data)

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [
                        torch.zeros_like(preds).to(self.device)
                        for _ in range(self.world_size)
                    ]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [
                        torch.zeros_like(preds_depth).to(self.device)
                        for _ in range(self.world_size)
                    ]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [
                        torch.zeros_like(truths).to(self.device)
                        for _ in range(self.world_size)
                    ]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(
                        self.workspace,
                        "validation",
                        f"{self.name}_{self.epoch:04d}_{self.local_step:04d}.png",
                    )
                    save_path_depth = os.path.join(
                        self.workspace,
                        "validation",
                        f"{self.name}_{self.epoch:04d}_{self.local_step:04d}_depth.png",
                    )
                    # save_path_gt = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch:04d}_{self.local_step:04d}_gt.png')

                    # self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(
                        save_path,
                        cv2.cvtColor(
                            (preds[0].detach().cpu().numpy() * 255).astype(np.uint8),
                            cv2.COLOR_RGB2BGR,
                        ),
                    )
                    cv2.imwrite(
                        save_path_depth,
                        (preds_depth[0].detach().cpu().numpy() * 255).astype(np.uint8),
                    )
                    # cv2.imwrite(save_path_gt, cv2.cvtColor((truths[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})"
                    )
                    pbar.update(1)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == "min" else -result
                )  # if max mode, use -result
            else:
                self.stats["results"].append(
                    average_loss
                )  # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, full=False, best=False):

        state = {
            "epoch": self.epoch,
            "stats": self.stats,
        }

        if self.model.cuda_ray:
            state["mean_count"] = self.model.mean_count
            state["mean_density"] = self.model.mean_density

        if full:
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
            state["scaler"] = self.scaler.state_dict()
            if self.ema is not None:
                state["ema"] = self.ema.state_dict()

        if not best:

            state["model"] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                if (
                    self.stats["best_result"] is None
                    or self.stats["results"][-1] < self.stats["best_result"]
                ):
                    self.log(
                        f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}"
                    )
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state["model"] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(
                    f"[WARN] no evaluated results found, skip saving best checkpoint."
                )

    def load_checkpoint(self, checkpoint=None):

        self.model, checkpoint_dict = load_model_checkpoint(
            self.model,
            self.ckpt_path,
            self.name,
            checkpoint=checkpoint,
            device=self.device,
            log=self.log,
        )

        if self.ema is not None and "ema" in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict["ema"])

        self.stats = checkpoint_dict["stats"]
        self.epoch = checkpoint_dict["epoch"]

        if self.model.cuda_ray:
            if "mean_count" in checkpoint_dict:
                self.model.mean_count = checkpoint_dict["mean_count"]
            if "mean_density" in checkpoint_dict:
                self.model.mean_density = checkpoint_dict["mean_density"]

        if self.optimizer and "optimizer" in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer, use default.")

        # strange bug: keyerror 'lr_lambdas'
        if self.lr_scheduler and "lr_scheduler" in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler, use default.")

        if "scaler" in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict["scaler"])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler, use default.")


def get_config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--workspace", type=str, default="workspace")
    parser.add_argument("--seed", type=int, default=0)
    ### training options
    parser.add_argument("--num_rays", type=int, default=4096)
    parser.add_argument(
        "--cuda_ray",
        action="store_true",
        help="use CUDA raymarching instead of pytorch",
    )
    # (only valid when not using --cuda_ray)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--upsample_steps", type=int, default=128)
    parser.add_argument("--max_ray_batch", type=int, default=4096)
    ### network backbone options
    parser.add_argument(
        "--fp16", action="store_true", help="use amp mixed precision training"
    )
    parser.add_argument("--ff", action="store_true", help="use fully-fused MLP")
    parser.add_argument("--tcnn", action="store_true", help="use TCNN backend")
    ### dataset options
    parser.add_argument(
        "--mode",
        type=str,
        default="colmap",
        help="dataset mode, supports (colmap, blender)",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="preload all data into GPU, fasten training but use more GPU memory",
    )
    # (default is for the fox dataset)
    parser.add_argument(
        "--bound",
        type=float,
        default=2,
        help="assume the scene is bounded in box(-bound, bound)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.33,
        help="scale camera location into box(-bound, bound)",
    )
    ### GUI options
    parser.add_argument("--gui", action="store_true", help="start a GUI")
    parser.add_argument("--W", type=int, default=800, help="GUI width")
    parser.add_argument("--H", type=int, default=800, help="GUI height")
    parser.add_argument(
        "--radius", type=float, default=5, help="default GUI camera radius from center"
    )
    parser.add_argument(
        "--fovy", type=float, default=90, help="default GUI camera fovy"
    )
    parser.add_argument(
        "--max_spp", type=int, default=64, help="GUI rendering max sample per pixel"
    )
    parser.add_argument(
        "--opt_poses",
        action="store_true",
        help="Flag to train camera poses in addition to NeRF params",
    )
    parser.add_argument(
        "--trans_noise",
        type=float,
        default=0.0,
        help="Variance of translation noise to add to GT poses.",
    )
    parser.add_argument(
        "--rot_noise",
        type=float,
        default=0.0,
        help="Variance of rotation noise to add to GT poses.",
    )

    return parser


def SE3_from_transform(T):
    """
    Constructs a lietorch.SE3 object from a batch of transform matrices.
    Args:
        T: batch of numpy transform matrices, [B, 4, 4].
    Returns a lietorch.SE3 object with the correct data.
    """
    # Check if we need to add a batch dim.
    if len(T.shape) < 3:
        T = T.view(1, 4, 4)

    q = Rotation.from_matrix(T[..., :3, :3]).as_quat()
    p = T[..., :3, -1]

    pose_data = torch.from_numpy(np.concatenate([p, q], -1)).float()

    return pose_data


def create_pose_var(dataset, requires_grad=False):
    pose_var = lietorch.LieGroupParameter(dataset.poses.cuda())
    pose_var.requires_grad = requires_grad

    return pose_var
