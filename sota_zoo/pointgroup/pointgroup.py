from typing import List
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from spconv.pytorch.utils import gather_features_by_pc_voxel_id
import pytorch_lightning as pl

from sota_zoo.metrics.segmentation import pixel_accuracy, mean_iou
from .unet import UNet, UBlock
from .datasets.scannet import PointCloud


class PointGroup(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: List[int],
        block_repeat: int = 2,
        learning_rate: float = 1e-3,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.in_channels = in_channels
        self.num_classes = num_classes

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.unet = UNet.build(in_channels, channels, block_repeat, norm_fn)
        self.sem_seg_head = nn.Linear(channels[0], num_classes)
        self.offset_head = nn.Sequential(
            nn.Linear(channels[0], channels[0]),
            norm_fn(channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[0], 3),
        )

        self.learning_rate = learning_rate

    def forward_proposal_score(self):
        ...

    def forward(
        self,
        x: spconv.SparseConvTensor,
        pc_voxel_id: torch.Tensor,
    ):
        x = self.unet(x)
        x = gather_features_by_pc_voxel_id(x.features, pc_voxel_id)

        sem_logits = self.sem_seg_head(x)
        # sem_preds = torch.argmax(sem_logits, dim=-1)

        pt_offsets = self.offset_head(x)

        return sem_logits, pt_offsets

    def collate(self, point_clouds: List[PointCloud]):
        # To GPU
        # point_clouds = [pc.to_tensor() for pc in point_clouds]

        points = torch.cat([pc.points for pc in point_clouds], dim=0)

        # Generate batch indices
        coord_dtype = point_clouds[0].voxel_coords[0].dtype
        batch_indices = torch.cat([
            torch.full((pc.voxel_coords.shape[0],), i, dtype=coord_dtype, device=self.device)
            for i, pc in enumerate(point_clouds)
        ], dim=0)

        voxel_coords = torch.cat([pc.voxel_coords for pc in point_clouds], dim=0)
        voxel_coords = torch.cat([batch_indices[:, None], voxel_coords], dim=-1)

        num_points_per_voxel = torch.cat([pc.num_points_per_voxel for pc in point_clouds], dim=0)
        voxel_features = torch.cat([pc.voxel_features for pc in point_clouds], dim=0)
        if self.in_channels == 3:
            voxel_features = voxel_features[:, :, :3]
        voxel_features = voxel_features.sum(1) / num_points_per_voxel[:, None]

        voxel_tensor = spconv.SparseConvTensor(
            voxel_features, voxel_coords,
            spatial_shape=(200, 400, 400),
            batch_size=len(point_clouds),
        )

        pc_voxel_id = []
        num_voxel_offset = 0
        for pc in point_clouds:
            pc.pc_voxel_id[pc.pc_voxel_id >= 0] += num_voxel_offset
            pc_voxel_id.append(pc.pc_voxel_id)
            num_voxel_offset += pc.voxel_coords.shape[0]
        pc_voxel_id = torch.cat(pc_voxel_id, dim=0)

        sem_labels = torch.cat([pc.sem_labels for pc in point_clouds], dim=0)

        instance_labels = torch.cat([pc.instance_labels for pc in point_clouds], dim=0)
        instance_regions = torch.cat([pc.instance_regions for pc in point_clouds], dim=0)

        return points, voxel_tensor, pc_voxel_id, sem_labels, instance_labels, instance_regions

    def _training_or_validation_step(
        self,
        point_clouds: List[PointCloud],
        batch_idx: int,
        training: bool,
    ):
        points, voxel_tensor, pc_voxel_id, sem_labels, instance_labels, instance_regions = self.collate(point_clouds)

        sem_logits, pt_offsets = self.forward(voxel_tensor, pc_voxel_id)

        valid_mask = pc_voxel_id >= 0
        assert valid_mask.shape[0] == sem_logits.shape[0]

        sem_logits = sem_logits[valid_mask]
        sem_labels = sem_labels[valid_mask]
        loss_sem_seg = F.cross_entropy(
            sem_logits, sem_labels, ignore_index=-100, reduction="mean"
        )

        gt_offsets = instance_regions[:, :3] - points[:, :3]
        pt_diff = pt_offsets - gt_offsets
        pt_dist = torch.sum(pt_diff.abs(), dim=-1)
        valid_mask = valid_mask & (instance_labels != -100)
        loss_pt_offset_dist = pt_dist[valid_mask].mean()

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=-1)
        gt_offsets = gt_offsets / (gt_offsets_norm[:, None] + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=-1)
        pt_offsets = pt_offsets / (pt_offsets_norm[:, None] + 1e-8)
        dir_diff = -(gt_offsets * pt_offsets).sum(-1)
        loss_pt_offset_dir = dir_diff[valid_mask].mean()

        loss = loss_sem_seg + loss_pt_offset_dist + loss_pt_offset_dir

        prefix = "train" if training else "val"
        batch_size = len(point_clouds)
        self.log(f"{prefix}_loss", loss, batch_size=batch_size)
        self.log(f"{prefix}/loss_sem_seg", loss_sem_seg, batch_size=batch_size)
        self.log(f"{prefix}/loss_pt_offset_dist", loss_pt_offset_dist, batch_size=batch_size)
        self.log(f"{prefix}/loss_pt_offset_dir", loss_pt_offset_dir, batch_size=batch_size)
        self.log(f"{prefix}/pixel_acc", pixel_accuracy(sem_logits, sem_labels), batch_size=batch_size)
        self.log(
            f"{prefix}/mean_iou",
            mean_iou(sem_logits, sem_labels, num_classes=self.num_classes),
            batch_size=batch_size
        )

        return sem_logits, pt_offsets, loss

    def training_step(self, point_clouds: List[PointCloud], batch_idx: int):
        _, _, loss = self._training_or_validation_step(
            point_clouds, batch_idx, training=True
        )

        return loss

    def validation_step(self, point_clouds: List[PointCloud], batch_idx: int):
        sem_logits, pt_offsets, _ = self._training_or_validation_step(
            point_clouds, batch_idx, training=False
        )


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
            amsgrad=True,
        )
        return optimizer
