import argparse
import os
from collections import defaultdict
from datetime import datetime
from typing import Optional

import copick
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
)
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from losses.custom_loss import CustomLoss
from metrics.czii2024_fbeta_metrics import CZII2024Metrics
from models import create_model
from utils import load_settings
from utils.config import get_config
from utils.slicing import extract_3d_patches_minimal_overlap, reconstruct_array

seed_everything(42, workers=True)
# Priotize performace over precision
torch.set_float32_matmul_precision("medium")
settings = load_settings()


# ## Define PyTorch lightning modules and the trainer
class Model(pl.LightningModule):
    def __init__(
        self,
        config: dict,
        # [1.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0]
    ):

        super().__init__()
        self.lr = config["optimizer"]["lr"]
        self.save_hyperparameters()
        self.config = config

        self.model = create_model(**config["model"])

        self.loss_fn = CustomLoss(**config["loss"])

        self.metric_fn = CZII2024Metrics(
            raw_data_dir=settings.raw_data_dir, **config["metric"]
        )

        self.train_loss = 0
        self.valid_loss = 0
        self.valid_tversky_loss = 0
        self.valid_ce_loss = 0
        self.val_metric = 0
        self.num_train_batch = 0
        self.num_val_batch = 0
        self.pred_masks = []
        self.out_channels = self.model.out_channels

    def forward(self, x):
        return self.model(x)

    def convert_labels(self, labels):

        num_classes = self.trainer.datamodule.nclasses

        labels_one_hot = F.one_hot(
            labels.squeeze(1).long(), num_classes=num_classes
        ).permute(0, 4, 1, 2, 3)

        # remove classes 2
        slices = [0, 1, 3, 4, 5, 6]
        labels_one_hot_sliced = labels_one_hot[:, slices]
        labels = labels_one_hot_sliced.argmax(1).unsqueeze(1).float()

        return labels

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        num_classes = self.trainer.datamodule.nclasses
        if self.out_channels != num_classes:
            y = self.convert_labels(y)

        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)["loss"]

        self.train_loss += loss
        self.num_train_batch += 1
        torch.cuda.empty_cache()
        return loss

    def on_train_epoch_end(self):
        loss_per_epoch = self.train_loss / self.num_train_batch
        self.log("train_loss", loss_per_epoch, prog_bar=True, sync_dist=True)

        self.train_loss = 0
        self.num_train_batch = 0

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():  # This ensures that gradients are not stored in memory
            x, y = batch["image"], batch["label"]

            num_classes = self.trainer.datamodule.nclasses
            if self.out_channels != num_classes:
                y = self.convert_labels(y)

            y_hat = self(x)

            loss_dict = self.loss_fn(y_hat, y)
            loss = loss_dict["loss"]

            metric_val_outputs = [
                AsDiscrete(argmax=True, to_onehot=self.out_channels)(i)
                for i in decollate_batch(y_hat)
            ]

            # construct outputs with the same shape with inputs for calculating f-beta score
            if self.out_channels != num_classes:

                insert_tensor = torch.zeros(
                    self.config["dataset"]["spatial_size"]
                ).unsqueeze(0).to(y_hat.device)
                
                metric_val_outputs = [
                    torch.cat(
                        [tensor[:1], insert_tensor, tensor[1:]]
                    )
                    for tensor in metric_val_outputs
                ]

            self.pred_masks += metric_val_outputs

            self.valid_loss += loss
            self.valid_tversky_loss += loss_dict["tversky_loss"]
            self.valid_ce_loss += loss_dict["ce_loss"]
            self.num_val_batch += 1
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):

        loss_per_epoch = self.valid_loss / self.num_val_batch
        tversky_loss_per_epoch = self.valid_tversky_loss / self.num_val_batch
        ce_loss_per_epoch = self.valid_ce_loss / self.num_val_batch

        self.log("valid_loss", loss_per_epoch, prog_bar=True, sync_dist=True)
        self.log(
            "valid_tversky_loss", tversky_loss_per_epoch, prog_bar=False, sync_dist=True
        )
        self.log("valid_ce_loss", ce_loss_per_epoch, prog_bar=False, sync_dist=True)

        tomo_size = (7, 184, 630, 630)
        valid_id = self.trainer.datamodule.valid_id
        patch_size = self.trainer.datamodule.spatial_size
        coordinates = self.trainer.datamodule.coordinates

        reconstructed_mask = reconstruct_array(
            self.pred_masks, coordinates, tomo_size, patch_size
        )

        gb, metric_per_epoch = self.metric_fn(reconstructed_mask, valid_id)

        self.log("val_metric", metric_per_epoch, prog_bar=True, sync_dist=True)

        if self.trainer.local_rank == 0:
            epoch = self.trainer.current_epoch
            iteration = self.trainer.global_step
            if isinstance(gb, int):
                desc = f"epoch: {epoch}, iteration: {iteration}, local lb: {metric_per_epoch}\n{gb}\n\n"
            else:
                desc = f"epoch: {epoch}, iteration: {iteration}, local lb: {metric_per_epoch}\n{gb.to_string()}\n\n"

            save_path = os.path.join(self.trainer.log_dir, "metrics.txt")
            with open(save_path, "a") as f:
                f.write(desc)

        self.valid_loss = 0
        self.valid_tversky_loss = 0
        self.valid_ce_loss = 0
        self.val_metric = 0
        self.num_val_batch = 0
        self.pred_masks = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


class CopickDataModule(pl.LightningDataModule):
    def __init__(self, copick_config_path: str, config: dict):

        super().__init__()
        self.train_batch_size = config["dataset"]["train_batch_size"]
        self.val_batch_size = config["dataset"]["val_batch_size"]
        self.spatial_size = config["dataset"]["spatial_size"]
        self.overlap_size = config["dataset"]["overlap_size"]
        self.valid_id = config["metric"]["valid_id"]
        self.num_random_samples_per_batch = config["dataset"][
            "num_random_samples_per_batch"
        ]
        self.save_hyperparameters()

        self.data_dicts, self.nclasses = self.data_from_copick(copick_config_path)

        self.train_files = [
            item for item in self.data_dicts if self.valid_id not in item["id"]
        ]
        self.val_files = [
            item
            for item in self.data_dicts
            if item["id"] == self.valid_id + "_denoised"
        ]
        print(f"Number of training samples: {len(self.train_files)}")
        print(f"Number of validation samples: {len(self.val_files)}")

        # Non-random transforms to be cached
        self.non_random_transforms = Compose(
            [
                EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
                NormalizeIntensityd(keys="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
            ]
        )

        # transforms to be applied during training
        self.random_transforms = Compose(
            [
                RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.spatial_size,
                    num_classes=self.nclasses,
                    ratios=[1, 1, 1, 1, 2, 1, 2],
                    num_samples=self.num_random_samples_per_batch,
                ),
                RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[1, 2]),
                RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=2),
                RandAffined(
                    keys=["image", "label"],
                    prob=0.3,
                    rotate_range=(0.1, 0.1, 0.1),
                    scale_range=(0.1, 0.1, 0.1),
                ),
                RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
                RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
            ]
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = CacheDataset(
            data=self.train_files, transform=self.non_random_transforms, cache_rate=1.0
        )
        self.train_ds = Dataset(data=self.train_ds, transform=self.random_transforms)

        patch_size_z, patch_size_x, patch_size_y = self.spatial_size
        overlap_z, overlap_x, overlap_y = self.overlap_size

        val_images = [dcts["image"] for dcts in self.val_files]
        val_labels = [dcts["label"] for dcts in self.val_files]
        val_image_patches, coordinates = extract_3d_patches_minimal_overlap(
            val_images,
            patch_size_z,
            patch_size_x,
            patch_size_y,
            overlap_z,
            overlap_x,
            overlap_y,
        )
        val_label_patches, coordinates = extract_3d_patches_minimal_overlap(
            val_labels,
            patch_size_z,
            patch_size_x,
            patch_size_y,
            overlap_z,
            overlap_x,
            overlap_y,
        )
        val_patched_data = [
            {"image": img, "label": lbl}
            for img, lbl in zip(val_image_patches, val_label_patches)
        ]
        self.val_ds = CacheDataset(
            data=val_patched_data, transform=self.non_random_transforms, cache_rate=1.0
        )
        self.coordinates = coordinates

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,  # Ensure the data order remains consistent
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

    @staticmethod
    def data_from_copick(
        copick_config_path,
        copick_user_name="copickUtils",
        copick_segmentation_name="paintedPicks",
        tomo_type="denoised",
    ):
        root = copick.from_file(copick_config_path)
        nclasses = len(root.pickable_objects) + 1
        data_dicts = []
        target_objects = defaultdict(dict)
        for object in root.pickable_objects:
            if object.is_particle:
                target_objects[object.name]["label"] = object.label
                target_objects[object.name]["radius"] = object.radius

        data_dicts = []
        for run in tqdm(root.runs):
            for tomo_type in ["denoised", "ctfdeconvolved", "isonetcorrected", "wbp"]:
                tomogram = run.get_voxel_spacing(10).get_tomogram(tomo_type).numpy()
                segmentation = run.get_segmentations(
                    user_id=copick_user_name,
                    name=copick_segmentation_name,
                    voxel_size=10,
                    is_multilabel=True,
                )[0].numpy()

                data_dicts.append(
                    {
                        "image": tomogram,
                        "label": segmentation,
                        "id": f"{run.name}_{tomo_type}",
                    }
                )

        return data_dicts, nclasses


def main(args: argparse.Namespace):

    # output directory
    args.dst_root = settings.model_checkpoint_dir
    args.dst_root.mkdir(parents=True, exist_ok=True)
    print(f"dst_root: {args.dst_root}")

    # config
    config = get_config(args.config, args.valid_id, args.options)
    print(yaml.dump(config))

    # tensorboard logging
    model_name = config["model"]["model_name"]
    spatial_size = "_".join(map(str, config["dataset"]["spatial_size"]))
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    valid_id = config["metric"]["valid_id"]

    exp_name = f"{model_name}_{spatial_size}_{current_time}"
    save_dir = args.dst_root / exp_name
    tb_logger = TensorBoardLogger(save_dir, name=valid_id)

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{step}-{valid_loss:.4f}-{val_metric:.4f}",
        save_top_k=5,
        monitor="val_metric",
        mode="max",
    )

    # learning rate callback
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # initilize model
    model = Model(config)

    # initilize data module
    copick_config_path = str(settings.train_data_working_dir / "copick.config")
    datamodule = CopickDataModule(copick_config_path, config)

    # Trainer for distributed training with DDP
    trainer = Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        **config["trainer"],
    )

    trainer.fit(model, datamodule=datamodule)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--valid_id", type=str, default=None)
    parser.add_argument("--options", type=str, nargs="*", default=list())
    args = parser.parse_args()

    print(f"config: {args.config}")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
