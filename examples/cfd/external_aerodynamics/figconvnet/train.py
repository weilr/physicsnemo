# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import logging
import logging.config
import os
import re
from timeit import default_timer
import webdataset as wds

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
# from tensorboardX import SummaryWriter

import torch
import torch.distributed as dist
import torch.utils
import torch.utils.data
import torchinfo

import warp as wp

from physicsnemo.distributed import DistributedManager

from src.utils import rank0
from src.utils.average_meter import AverageMeter, AverageMeterDict, Timer
from src.utils.loggers import init_logger
from src.utils.seed import set_seed
from src.utils.signal_handlers import SignalHandler
from src.utils.early_stopping import EarlyStopping
from physicsnemo.models.figconvnet.geometries import GridFeaturesMemoryFormat


logger = logging.getLogger("figconv")


def _delete_previous_checkpoints(config):
    checkpoints_to_delete = []
    for f in os.listdir(config.output):
        if re.compile(r"^model_(\d{5})\.pth$").match(f):
            checkpoints_to_delete.append(f)
    checkpoints_to_delete.sort()
    checkpoints_to_delete = checkpoints_to_delete[: -config.train.num_checkpoints]
    logger.info(f"Deleting {len(checkpoints_to_delete)} checkpoints")
    for f in checkpoints_to_delete:
        try:
            os.remove(os.path.join(config.output, f))
        except FileNotFoundError:
            pass


@rank0
def _save_state(model, optimizer, scheduler, scaler, epoch, tot_iter, config, file_name=None):
    save_path = os.path.join(config.output, f"model_{epoch:05d}.pth" if file_name is None else file_name)
    logger.info(f"Saving model at epoch {epoch} to {save_path}")
    state_dict = {
        "model": model.model().state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "tot_iter": tot_iter,
    }
    # Save the file with 0000X format
    torch.save(state_dict, save_path)
    _delete_previous_checkpoints(config)

def _load_model_state(model, checkpoint_path):
    # Get rank if distributed
    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    map_location = {"cuda:0": f"cuda:{rank}"}
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    model.model().load_state_dict(checkpoint["model"])
    start_epoch = checkpoint["epoch"] + 1
    tot_iter = checkpoint["tot_iter"]
    
    # Wait until all processes load the checkpoint
    if DistributedManager().distributed:
        torch.distributed.barrier()

    return start_epoch, tot_iter

def _resume_from_checkpoint(model, optimizer, scheduler, scaler, config):
    logger.info(f"Resuming from {config.output}")

    # Find the latest checkpoint
    checkpoints = []
    for f in os.listdir(config.output):
        if f.startswith("model_") and f.endswith(".pth"):
            checkpoints.append(f)
    checkpoints.sort()

    start_epoch = 0
    tot_iter = 0
    # Load if there is a checkpoint
    if len(checkpoints) == 0:
        logger.info("No checkpoints found")
    else:
        logger.info(f"Found {len(checkpoints)} checkpoints")
        logger.info(f"Loading {checkpoints[-1]}")
        checkpoint_path = os.path.join(config.output, checkpoints[-1])
        # Get rank if distributed
        rank = 0
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        map_location = {"cuda:0": f"cuda:{rank}"}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        model.model().load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        tot_iter = checkpoint["tot_iter"]

        # Wait until all processes load the checkpoint.
        if DistributedManager().distributed:
            torch.distributed.barrier()

    return start_epoch, tot_iter


@torch.no_grad()
def val_model(model, datamodule, config, autocast, loss_fn=None):
    model.eval()
    val_loader = datamodule.val_dataloader(
        batch_size=config.train.batch_size, **config.train.dataloader
    )

    val_meter = AverageMeterDict()
    for i, data_dict in enumerate(val_loader):
        with autocast():
            loss_dict = model.val_dict(
                data_dict, config, loss_fn=loss_fn, datamodule=datamodule
            )
            val_meter.update(loss_dict)
        if i % config.train.print_interval == 0:
            print_str = f"[Validation] Iter {i}: "
            for k, v in val_meter.avg.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                if isinstance(v, float):
                    print_str += f"{k}: {v:.8f}, "
                else:
                    print_str += f"{k}: {v}, "
            logger.info(print_str)
    
    val_meter.all_gather_attributes()

    model.train()
    return val_meter.avg


@torch.no_grad()
def test_model(model, datamodule, config, loss_fn=None):
    model.eval()
    test_loader = datamodule.test_dataloader(
        batch_size=config.eval.batch_size, **config.eval.dataloader
    )
    eval_meter = AverageMeterDict()
    visualize_data_dicts = []
    eval_timer = Timer()
    for i, data_dict in enumerate(test_loader):
        eval_timer.tic()
        out_dict = model.eval_dict(data_dict, loss_fn=loss_fn, datamodule=datamodule)
        out_dict["inference_time"] = eval_timer.toc()
        eval_meter.update(out_dict)
        if i % config.eval.plot_interval == 0:
            visualize_data_dicts.append(data_dict)
        if i % config.eval.print_interval == 0:
            # Print eval dict
            print_str = f"[Test] Eval {i}: "
            for k, v in eval_meter.avg.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                if isinstance(v, float):
                    print_str += f"{k}: {v:.8f}, "
                else:
                    print_str += f"{k}: {v}, "
            logger.info(print_str)

    # Merge all dictionaries
    merged_image_dict = {}
    merged_point_cloud_dict = {}
    if hasattr(model, "image_pointcloud_dict"):
        for i, data_dict in enumerate(visualize_data_dicts):
            image_dict, pointcloud_dict = model.image_pointcloud_dict(
                data_dict, datamodule=datamodule
            )
            for k, v in image_dict.items():
                merged_image_dict[f"{k}_{i}"] = v
            for k, v in pointcloud_dict.items():
                merged_point_cloud_dict[f"{k}_{i}"] = v
    elif hasattr(model, "image_dict"):
        for i, data_dict in enumerate(visualize_data_dicts):
            image_dict, pointcloud_dict = model.image_dict(
                data_dict, datamodule=datamodule
            )
            for k, v in image_dict.items():
                merged_image_dict[f"{k}_{i}"] = v

    # Aggregate all counts, sums, avgs, and private attributes if distributed
    eval_meter.all_gather_attributes()

    eval_dict = eval_meter.avg

    # Post process the eval dict
    if hasattr(model, "post_eval_epoch"):
        (
            eval_dict,
            merged_image_dict,
            merged_point_cloud_dict,
        ) = model.post_eval_epoch(
            eval_dict,
            merged_image_dict,
            merged_point_cloud_dict,
            eval_meter._private_attributes,
            datamodule,
        )

    model.train()
    return eval_dict, merged_image_dict, merged_point_cloud_dict


def train(config: DictConfig, signal_handler: SignalHandler):
    dist = DistributedManager()

    # Initialize the device. Allow device override only in non-distributed setting.
    device = dist.device if dist.distributed else torch.device(config.device)
    # Set default devices.
    torch.cuda.device(device)
    wp.init()
    wp.set_device(str(device))

    loggers = init_logger(config)
    logger.info(f"Config summary:\n{OmegaConf.to_yaml(config, sort_keys=True)}")

    # Initialize the model
    model = instantiate(config.model)
    model = model.to(device)
    # Print model summary (structure and parmeter count).
    logger.info(f"Model summary:\n{torchinfo.summary(model, verbose=0)}\n")

    # Enable DDP.
    if dist.distributed:
        # TODO(akamenev): make broadcast_buffers configurable
        # since some of the models use BatchNorm.
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.device],
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
        )
        logger.info("Initialized DDP.")
        # 不确定是否有效
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Set the original model getter to simplify access.
    assert not hasattr(model, "model")
    type(model).model = (lambda m: m.module) if dist.distributed else (lambda m: m)

    # Initialize the dataloaders
    datamodule = instantiate(config.data)
    train_loader = datamodule.train_dataloader(
        batch_size=config.train.batch_size, **config.train.dataloader
    )

    # Initialize the optimizer and scheduler.
    optimizer = instantiate(config.optimizer, model.parameters())
    scheduler = instantiate(config.lr_scheduler, optimizer)

    # Initialize the loss function.
    loss_fn = instantiate(config.loss)
    if config.eval.loss is None:
        eval_loss_fn = loss_fn
    else:
        eval_loss_fn = instantiate(config.eval.loss)

    # Initialize AMP.
    scaler = instantiate(config.amp.scaler)
    autocast = partial(
        torch.cuda.amp.autocast,
        enabled=config.amp.enabled,
        dtype=hydra.utils.get_object(config.amp.autocast.dtype),
    )

    # Resume if resume is True
    start_epoch = 0
    tot_iter = 0
    if config.train.resume and os.path.exists(config.output):
        start_epoch, tot_iter = _resume_from_checkpoint(
            model, optimizer, scheduler, scaler, config
        )

    if config.test_best_only:
        best_model_path = os.path.join(config.output, "best_model.pth")
        if os.path.exists(best_model_path):
            logger.info(f"Loading best model: {best_model_path}")
            best_epoch, best_iter = _load_model_state(model, best_model_path)
        else:
            logger.info("No best model found")
            return
        
        # Run final evaluation
        eval_dict, eval_images, eval_point_clouds = test_model(
            model.model(), datamodule, config, eval_loss_fn
        )
        for k, v in eval_dict.items():
            logger.info(f"[Final Test with Best Model] Epoch: {best_epoch} Iter: {best_iter} {k}: {v:.8f}")
        return

    # Eval first for debugging
    if config.eval.run_eval_first:
        eval_dict, eval_images, eval_point_clouds = test_model(
            model.model(), datamodule, config, eval_loss_fn
        )
        for k, v in eval_dict.items():
            logger.info(f"First Eval: {k}: {v:.8f}")

    # Initialize EarlyStopping if enabled
    if config.train.early_stopping.enabled and dist.rank == 0:
        # Get metric weights from config or use default
        metric_weights = getattr(config.train.early_stopping, "metric_weights", {"val_loss": 1.0})
        mode = getattr(config.train.early_stopping, "mode", "min")
        
        early_stopping = EarlyStopping(
            patience=config.train.early_stopping.patience,
            delta=config.train.early_stopping.delta,
            verbose=config.train.early_stopping.verbose,
            save_state_fn=_save_state,
            logger=logger,
            metric_weights=metric_weights,
            mode=mode,
        )

    # cnt = 0
    end_epoch = config.train.num_epochs-1
    for ep in range(start_epoch, config.train.num_epochs):
        model.train()
        t1 = default_timer()
        train_l2_meter = AverageMeter()

        datamodule.set_epoch(train_loader, ep)

        for data_dict in train_loader:
            # Check if the signal is received
            if signal_handler.is_stopped():
                logger.debug("Signal received. Breaking the training loop.")
                break

            optimizer.zero_grad()

            with autocast():
                # if cnt == 0:
                #     writer = SummaryWriter('struct/model_visualization')
                #     with torch.no_grad():
                #         # Set model to eval mode temporarily
                #         training = model.model().training
                #         model.model().eval()
                        
                #         # Fix all random states
                #         torch.manual_seed(0)
                #         torch.cuda.manual_seed(0)
                #         torch.backends.cudnn.deterministic = True
                #         torch.backends.cudnn.benchmark = False
                        
                #         # Prepare dummy input with fixed shape and values
                #         dummy_input = data_dict["cell_centers"].float().to(device)
                #         dummy_input = dummy_input[:1].detach().clone()  # Use only first batch
                #         dummy_input.requires_grad_(False)
                        
                #         try:
                #             # Try to use scripting first
                #             scripted_model = None
                #             try:
                #                 scripted_model = torch.jit.script(model.model())
                #                 writer.add_graph(scripted_model, dummy_input)
                #             except Exception as e:
                #                 logger.warning(f"Failed to script model: {e}")
                                
                #             # If scripting fails, fall back to tracing
                #             if scripted_model is None:
                #                 with torch.jit.optimized_execution(False):
                #                     traced_model = torch.jit.trace(model.model(), dummy_input, check_trace=False)
                #                     writer.add_graph(traced_model, dummy_input)
                                    
                #         finally:
                #             # Restore model's training mode and cudnn settings
                #             model.model().train(training)
                #             torch.backends.cudnn.deterministic = False
                #             torch.backends.cudnn.benchmark = True
                            
                #     writer.close()
                #     cnt += 1
                loss_dict = model.model().loss_dict(
                    data_dict, loss_fn=loss_fn, datamodule=datamodule
                )

            loss = 0
            for k, v in loss_dict.items():
                v = v * getattr(config, k + "_weight", 1)
                loss = loss + v.mean()

            # Assert loss is valid
            assert torch.isfinite(loss).all(), f"Loss is not finite: {loss}"

            # Note: if AMP is disabled, the scaler will fall back to the default behavior.
            scaler.scale(loss).backward()

            # TODO(akamenev): grad clipping can be used not only in AMP.
            if config.amp.clip_grad:
                # Unscales the gradients of optimizer's assigned params in-place.
                scaler.unscale_(optimizer)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.amp.grad_max_norm
                )

            # If optimizer's gradients were already unscaled, the scaler.step does not unscale them,
            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            train_l2_meter.update(loss.item())
            loggers.log_scalar("train/iter_lr", scheduler.get_last_lr()[0], tot_iter)
            loggers.log_scalar("train/iter_loss", loss.item(), tot_iter)
            for k, v in loss_dict.items():
                loggers.log_scalar(f"train/{k}", v.item(), tot_iter)
            if tot_iter % config.train.print_interval == 0:
                print_str = f"[Train] Iter {tot_iter} loss: {loss.item():.8f}, "
                for k, v in loss_dict.items():
                    print_str += f"{k}: {v.item():.8f}, "  # only print the number
                logger.info(print_str)

            if config.train.lr_scheduler_mode == "iteration":
                scheduler.step()
            tot_iter += 1
            torch.cuda.empty_cache()

        t2 = default_timer()
        logger.info(
            f"Training epoch {ep} took {t2 - t1:.2f} seconds. L2 loss: {train_l2_meter.avg:.8f}"
        )
        loggers.log_scalar("train/epoch_train_l2", train_l2_meter.avg, tot_iter)
        loggers.log_scalar("train/train_epoch_duration", t2 - t1, tot_iter)
        loggers.log_scalar("train/epoch", ep, tot_iter)

        # Save the weights, optimization state, and scheduler state into one file
        if ep % config.train.save_interval == 0 or signal_handler.is_stopped():
            # save the model
            _save_state(model, optimizer, scheduler, scaler, ep, tot_iter, config)
        
        # Validation
        t1 = default_timer()
        val_dict = val_model(model.model(), datamodule, config, autocast, loss_fn=loss_fn)
        t2 = default_timer()
        val_loss = val_dict["val_loss"]
        logger.info(f"Validation epoch {ep} took {t2 - t1:.2f} seconds. Validation Loss: {val_loss:.8f}")
        loggers.log_scalar("val/loss", val_loss, tot_iter)

        old_lr =  scheduler.get_last_lr()[0]
        if config.train.lr_scheduler_mode == "epoch":
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                val_loss_tensor = torch.tensor(val_loss, device="cuda")
                torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.AVG)
                scheduler.step(val_loss_tensor.item())
            else:
                scheduler.step()
        new_lr =  scheduler.get_last_lr()[0]
        if new_lr != old_lr:
            logger.info(f'Learning rate decreased from {old_lr:.6f} to {new_lr:.6f}')

        # Simple early stopping based on validation loss only (backward compatibility)
        if config.train.early_stopping.enabled and  dist.rank == 0:
            es_metrics = {}
            for k, v in val_dict.items():
                if k in config.train.early_stopping.metric_weights:
                    es_metrics[k] = v
            
            early_stopping(es_metrics, 
                model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                scaler=scaler, 
                epoch=ep, 
                tot_iter=tot_iter, 
                config=config, 
                file_name=f"best_model.pth"
            )
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered in epoch {ep} with validation loss, exiting training loop...")
                signal_handler.stop()
                
        # Make sure all processes check is_stopped() to sync the stop signal
        if signal_handler.is_stopped():
            logger.info(f"Process {dist.rank} received stop signal")
            # Load best model before final evaluation
            best_model_path = os.path.join(config.output, "best_model.pth")
            if os.path.exists(best_model_path):
                logger.info("Loading best model for final evaluation")
                best_epoch, best_iter = _load_model_state(model, best_model_path)
            
            # Run final evaluation
            eval_dict, eval_images, eval_point_clouds = test_model(
                model.model(), datamodule, config, eval_loss_fn
            )
            for k, v in eval_dict.items():
                logger.info(f"[Final Test with Best Model] Epoch: {best_epoch} Iter: {best_iter} {k}: {v:.8f}")
            break
                
        # Regular evaluation
        if (
            ep % config.eval.interval == 0
            or ep == config.train.num_epochs - 1
            and (not signal_handler.is_stopped())
        ):
            eval_dict, eval_images, eval_point_clouds = test_model(
                model.model(), datamodule, config, eval_loss_fn
            )
            for k, v in eval_dict.items():
                logger.info(f"[Test] Epoch: {ep} {k}: {v:.8f}")
                loggers.log_scalar(f"eval/{k}", v, tot_iter)
            for k, v in eval_images.items():
                loggers.log_image(f"eval_vis/{k}", v, tot_iter)
            if config.log_pointcloud:
                for k, v in eval_point_clouds.items():
                    loggers.log_pointcloud(
                        f"eval_vis/{k}", v[..., :3], v[..., 3:], tot_iter
                    )
        end_epoch = ep

    # Save the final model if the training loop was not stopped by the signal handler.
    if not signal_handler.is_stopped():
        _save_state(
            model,
            optimizer,
            scheduler,
            scaler,
            end_epoch,
            tot_iter,
            config,
        )


def _slurm_setup(config: DictConfig) -> None:
    # Hydra config contains properly resolved absolute path.
    config.output = HydraConfig.get().runtime.output_dir

    # Detect if it is running on a SLURM cluster.
    if "SLURM_JOB_ID" in os.environ:
        # The output directory is set to simply ${output}/SLURM_JOB_ID.
        # config.output = os.path.join(config.output, os.environ["SLURM_JOB_ID"])
        # Check for the checkpoints and model_*.pth files in the output directory.
        if os.path.exists(config.output) and any(
            f.startswith("model_") and f.endswith(".pth")
            for f in os.listdir(config.output)
        ):
            config.train.resume = True


def _init_python_logging(config: DictConfig) -> None:
    if config.log_dir is None:
        config.log_dir = config.output
    else:
        config.log_dir = to_absolute_path(config.log_dir)

    # Make the log dir
    os.makedirs(config.log_dir, exist_ok=True)

    # Set up Python loggers.
    if pylog_cfg := OmegaConf.select(config, "logging.python"):
        pylog_cfg.output = config.output
        pylog_cfg.rank = DistributedManager().rank
        # Enable logging only on rank 0, if requested.
        if pylog_cfg.rank0_only and pylog_cfg.rank != 0:
            pylog_cfg.handlers = {}
            pylog_cfg.loggers.figconv.handlers = []
        # Configure logging.
        logging.config.dictConfig(OmegaConf.to_container(pylog_cfg, resolve=True))


@hydra.main(version_base="1.3", config_path="configs", config_name="base")
def main(config: DictConfig):
    _slurm_setup(config)

    _init_python_logging(config)

    # Set the random seed.
    if config.seed is not None:
        set_seed(config.seed)

    with SignalHandler(status_path=config.signal_handler.status_path) as signal_handler:
        t1 = default_timer()
        train(config, signal_handler)
        t2 = default_timer()
        logger.info(f"Training took {(t2-t1)/3600:.2f}hrs")


def _init_hydra_resolvers():
    def res_mem_pair(
        fmt: str, dims: list[int, int, int]
    ) -> tuple[GridFeaturesMemoryFormat, tuple[int, int, int]]:
        return getattr(GridFeaturesMemoryFormat, fmt), tuple(dims)

    OmegaConf.register_new_resolver("res_mem_pair", res_mem_pair)


if __name__ == "__main__":
    DistributedManager.initialize()

    _init_hydra_resolvers()

    main()
