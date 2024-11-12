# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Training script for GAN-based vocoders.

This module implements a Trainer class for training GAN-based vocoders. The training process includes
both generator and discriminator components, and provides methods for saving/loading model checkpoints
and evaluating model performance.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
"""

import os
import sys
from collections import defaultdict
from logging import getLogger
from typing import Dict, List, Tuple

import hydra
import librosa.display
import matplotlib
import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from joblib import load
from numpy import ndarray
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.utils.data import DataLoader

from wavehax.datasets import AudioFeatDataset

# Set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


# A logger for this file
logger = getLogger(__name__)


class Trainer:
    """Customized trainer module for GAN-based vocoder training."""

    def __init__(
        self,
        cfg,
        steps,
        epochs,
        data_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        device=torch.device("cpu"),
    ) -> None:
        """Initialize Trainer.

        Args:
            cfg (dict): Configuration dictionary loaded from a YAML file.
            steps (int): Initial count of global training steps.
            epochs (int): Initial count of global training epochs.
            data_loader (dict): Dictionary of data loaders containing "train" and "dev" loaders.
            model (dict): Dictionary of models containing "generator" and "discriminator" models.
            criterion (dict): Dictionary of loss functions.
            optimizer (dict): Dictionary of optimizers for the generator and discriminator.
            scheduler (dict): Dictionary of schedulers for the generator and discriminator.
            device (torch.device): Pytorch device instance (default: CPU).
        """
        self.cfg = cfg
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.finish_train = False
        self.writer = SummaryWriter(cfg.out_dir)
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

    def run(self) -> None:
        """Execute the training loop."""
        while True:
            # Train one epoch
            self._train_epoch()

            # Check whether training is finished
            if self.finish_train:
                break

        logger.info("Finished training.")

    def save_checkpoint(self, ckpt_path: str) -> None:
        """Save the current state of the model to a checkpoint.

        Args:
            ckpt_path (str): Path to save the checkpoint file.
        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if hasattr(self.model["generator"], "module"):
            state_dict["model"] = {
                "generator": self.model["generator"].module.state_dict(),
                "discriminator": self.model["discriminator"].module.state_dict(),
            }
        else:
            state_dict["model"] = {
                "generator": self.model["generator"].state_dict(),
                "discriminator": self.model["discriminator"].state_dict(),
            }

        if not os.path.exists(os.path.dirname(ckpt_path)):
            os.makedirs(os.path.dirname(ckpt_path))
        torch.save(state_dict, ckpt_path)

    def load_checkpoint(self, ckpt_path: str, load_only_params: bool = False) -> None:
        """Load the model state from a checkpoint.

        Args:
            ckpt_path (str): Path to the checkpoint file to be loaded.
            load_only_params (bool): If True, only model parameters are loaded, ignoring optimizers and schedulers.
        """
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.model["generator"].load_state_dict(state_dict["model"]["generator"])
        self.model["discriminator"].load_state_dict(
            state_dict["model"]["discriminator"]
        )
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(
                state_dict["optimizer"]["generator"]
            )
            self.optimizer["discriminator"].load_state_dict(
                state_dict["optimizer"]["discriminator"]
            )
            self.scheduler["generator"].load_state_dict(
                state_dict["scheduler"]["generator"]
            )
            self.scheduler["discriminator"].load_state_dict(
                state_dict["scheduler"]["discriminator"]
            )

    def _train_step(self, batch: Tuple) -> None:
        """Perform a single training step on a given batch.

        Args:
            batch (Tuple): A tuple containing batched data.
        """
        # Parse batch
        batch = [x.to(self.device) if x is not None else x for x in batch]
        y, cond, f0 = batch

        # Perform forward propagation
        y_, _ = self.model["generator"](cond, f0)

        # Calculate mel spectral loss
        mel_loss = self.criterion["mel"](y_, y)
        gen_loss = self.cfg.train.lambda_mel * mel_loss
        self.total_train_loss["train/mel_loss"] += mel_loss.item()

        # Calculate discriminator related losses
        if self.steps > self.cfg.train.discriminator_train_start_steps:
            # Calculate feature matching loss
            p_fake, fmaps_fake = self.model["discriminator"](y_)
            with torch.no_grad():
                p_real, fmaps_real = self.model["discriminator"](y)
            fm_loss = self.criterion["fm"](fmaps_fake, fmaps_real)
            gen_loss += self.cfg.train.lambda_fm * fm_loss
            self.total_train_loss["train/fm_loss"] += fm_loss.item()

            # Calculate adversarial loss
            adv_loss = self.criterion["adv"](p_fake)
            gen_loss += self.cfg.train.lambda_adv * adv_loss
            self.total_train_loss["train/adv_loss"] += adv_loss.item()

        self.total_train_loss["train/generator_loss"] += gen_loss.item()

        # Update generator
        self.optimizer["generator"].zero_grad()
        gen_loss.backward(retain_graph=True)
        if self.cfg.train.generator_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.cfg.train.generator_grad_norm,
            )
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()

        # Discriminator
        if self.steps > self.cfg.train.discriminator_train_start_steps:
            # Calculate discriminator loss
            p_fake, _ = self.model["discriminator"](y_.detach())
            p_real, _ = self.model["discriminator"](y)
            # NOTE: the first argument must to be the fake samples
            fake_loss, real_loss = self.criterion["adv"](p_fake, p_real)
            dis_loss = fake_loss + real_loss
            self.total_train_loss["train/fake_loss"] += fake_loss.item()
            self.total_train_loss["train/real_loss"] += real_loss.item()
            self.total_train_loss["train/discriminator_loss"] += dis_loss.item()

            # Update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.cfg.train.discriminator_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.cfg.train.discriminator_grad_norm,
                )
            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()

        self.total_train_loss["train/lr_g"] += self.scheduler[
            "generator"
        ].get_last_lr()[0]
        self.total_train_loss["train/lr_d"] += self.scheduler[
            "discriminator"
        ].get_last_lr()[0]

        # Update counts
        self.steps += 1
        self._check_train_finish()

    def _train_epoch(self) -> None:
        """Train the model for one complete epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # Train one step
            self._train_step(batch)

            # Check interval
            self._check_log_interval()
            self._check_eval_interval()
            self._check_save_interval()

            # Check whether training is finished
            if self.finish_train:
                return

        # Update counts
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logger.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

    @torch.no_grad()
    def _eval_step(self, batch: Tuple) -> None:
        """Perform a single evaluation step on a given batch.

        Args:
            batch (Tuple): A tuple containing batched data.
        """
        # Parse batch
        batch = [x.to(self.device) if x is not None else x for x in batch]
        y, c, f0 = batch

        # Perform forward propagation
        y_, p = self.model["generator"](c, f0)

        # Calculate mel spectral loss
        mel_loss = self.criterion["mel"](y_, y)
        gen_loss = self.cfg.train.lambda_mel * mel_loss
        self.total_eval_loss["eval/mel_loss"] += mel_loss.item()

        # Calculate discriminator related losses
        if self.steps > self.cfg.train.discriminator_train_start_steps:
            # Calculate feature matching loss
            p_fake, fmaps_fake = self.model["discriminator"](y_)
            p_real, fmaps_real = self.model["discriminator"](y)
            fm_loss = self.criterion["fm"](fmaps_fake, fmaps_real)
            gen_loss += self.cfg.train.lambda_fm * fm_loss
            self.total_eval_loss["eval/fm_loss"] += fm_loss.item()

            # Calculate adversarial loss
            adv_loss = self.criterion["adv"](p_fake)
            gen_loss += self.cfg.train.lambda_adv * adv_loss
            self.total_eval_loss["eval/adv_loss"] += adv_loss.item()

        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()

        # Discriminator
        if self.steps > self.cfg.train.discriminator_train_start_steps:
            # NOTE: the first augment must to be the fake sample
            fake_loss, real_loss = self.criterion["adv"](p_fake, p_real)
            dis_loss = fake_loss + real_loss
            self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
            self.total_eval_loss["eval/real_loss"] += real_loss.item()
            self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()

    def _eval_epoch(self) -> None:
        """Evaluate model one epoch."""
        logger.info(f"(Steps: {self.steps}) Start evaluation.")
        # Change to evaluation mode
        for key in self.model.keys():
            self.model[key].eval()

        # Calculate losses for each batch
        for eval_steps_per_epoch, batch in enumerate(self.data_loader["valid"], 1):
            # Evaluate one step
            self._eval_step(batch)

            # Save intermediate results
            if eval_steps_per_epoch == 1:
                self._generate_and_save_intermediate_result(batch)
            if eval_steps_per_epoch == 10:
                break

        logger.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # Calculate the averaged losses
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logger.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )

        # Record results to Tensorboard
        self._write_to_tensorboard(self.total_eval_loss)

        # Reset accumulated losses
        self.total_eval_loss = defaultdict(float)

        # Restore training mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _generate_and_save_intermediate_result(self, batch: Tuple) -> None:
        """Generate and save intermediate results during training.

        This method performs intermediate result visualization and logging during the training process,
        including waveform and spectrogram plots, and saving the corresponding audio.

        Args:
            batch (Tuple): A tuple containing batched data.
        """
        # Delayed import to avoid backend-related issues
        import matplotlib.pyplot as plt

        # Use only the first sample in the batch
        batch = [x[:1].to(self.device) if x is not None else x for x in batch]
        y, c, f0 = batch

        # Perform forward propagation
        y_, p = self.model["generator"](c, f0)

        # Visualize and save the real waveform
        sample_rate = self.model["generator"].sample_rate
        audio = y.view(-1).cpu().numpy()
        fig = plt.figure(figsize=(6, 3))
        plt.plot(audio[: int(sample_rate * 0.1)], linewidth=1)
        self.writer.add_figure("real/waveform", fig, self.steps)
        plt.close()
        fig = plt.figure(figsize=(8, 6))
        spectrogram = np.abs(
            librosa.stft(
                y=audio,
                n_fft=1024,
                hop_length=128,
                win_length=1024,
                window="hann",
            )
        )
        spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
        librosa.display.specshow(
            spectrogram_db,
            sr=sample_rate,
            hop_length=128,
            win_length=1024,
            y_axis="linear",
            x_axis="time",
        )
        self.writer.add_figure("real/spectrogram", fig, self.steps)
        plt.close()
        self.writer.add_audio(
            "real.wav",
            audio,
            self.steps,
            sample_rate,
        )

        # Visualize and save the fake waveform
        audio = y_.view(-1).cpu().numpy()
        fig = plt.figure(figsize=(6, 3))
        plt.plot(audio[: int(sample_rate * 0.1)], linewidth=1)
        self.writer.add_figure("fake/waveform", fig, self.steps)
        plt.close()
        fig = plt.figure(figsize=(8, 6))
        spectrogram = np.abs(
            librosa.stft(
                y=audio,
                n_fft=1024,
                hop_length=128,
                win_length=1024,
                window="hann",
            )
        )
        spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
        librosa.display.specshow(
            spectrogram_db,
            sr=sample_rate,
            hop_length=128,
            win_length=1024,
            y_axis="linear",
            x_axis="time",
        )
        self.writer.add_figure("fake/spectrogram", fig, self.steps)
        plt.close()
        self.writer.add_audio(
            "fake.wav",
            audio,
            self.steps,
            sample_rate,
        )

        # Visualize and save the prior waveform (for debug)
        if p is not None:
            audio = p.view(-1).cpu().numpy()
            fig = plt.figure(figsize=(6, 3))
            plt.plot(audio[: int(sample_rate * 0.1)], linewidth=1)
            self.writer.add_figure("prior/waveform", fig, self.steps)
            plt.close()
            fig = plt.figure(figsize=(8, 6))
            spectrogram = np.abs(
                librosa.stft(
                    y=audio,
                    n_fft=1024,
                    hop_length=128,
                    win_length=1024,
                    window="hann",
                )
            )
            spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
            librosa.display.specshow(
                spectrogram_db,
                sr=sample_rate,
                hop_length=128,
                win_length=1024,
                y_axis="linear",
                x_axis="time",
            )
            self.writer.add_figure("prior/spectrogram", fig, self.steps)
            plt.close()
            self.writer.add_audio(
                "prior.wav",
                audio,
                self.steps,
                sample_rate,
            )

    def _write_to_tensorboard(self, loss: Dict) -> None:
        """Write training loss metrics to TensorBoard.

        Args:
            loss (Dict): Dictionary containing loss metrics.
        """
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self) -> None:
        """Check if it's time to save a checkpoint."""
        if self.steps % self.cfg.train.save_interval_steps == 0:
            self.save_checkpoint(
                os.path.join(
                    self.cfg.out_dir,
                    "checkpoints",
                    f"checkpoint-{self.steps}steps.pkl",
                )
            )
            logger.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self) -> None:
        """Check if it's time to evaluate the model."""
        if self.steps % self.cfg.train.eval_interval_steps == 0:
            self._eval_epoch()

    def _check_log_interval(self) -> None:
        """Check if it's time to log training metrics."""
        if self.steps % self.cfg.train.log_interval_steps == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.cfg.train.log_interval_steps
                logger.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # Reset accumulated losses for the next logging interval
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self) -> None:
        """Check if training should be finished."""
        if self.steps >= self.cfg.train.train_max_steps:
            self.finish_train = True


class Collater:
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(
        self, sample_rate: int, hop_length: int, batch_max_length: int
    ) -> None:
        """Initialize the customized collater.

        Args:
            sample_rate (int): Sampling frequency.
            hop_length (int): Hop size for auxiliary features.
            batch_max_length (int): Maximum length of the batch.
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        # Adjust batch_max_length to be a multiple of hop_length
        if batch_max_length % hop_length != 0:
            batch_max_length += -(batch_max_length % hop_length)
        self.batch_max_length = batch_max_length
        self.batch_max_frames = batch_max_length // hop_length

    def __call__(self, batch: List) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert a list of audio-feature pairs into batched tensors.

        Args:
            batch (List): List of tuples containing audio, features, and F0 sequences.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Batched tensors for audio, features, and F0.
        """
        audio_batch, cond_batch, f0_batch = [], [], []
        for idx in range(len(batch)):
            audio, cond, f0 = batch[idx]
            if len(cond) > self.batch_max_frames:
                # Randomly pickup a segment with the specific length
                start_frame = np.random.randint(0, len(cond) - self.batch_max_frames)
                start_step = start_frame * self.hop_length
                audio = audio[start_step : start_step + self.batch_max_length]
                cond = cond[start_frame : start_frame + self.batch_max_frames]
                if f0 is not None:
                    f0 = f0[start_frame : start_frame + self.batch_max_frames]
                self._check_length(audio, cond, f0)
            else:
                logger.warn(f"Removed short sample from batch (length={len(cond)}).")
                continue
            audio_batch += [audio.reshape(1, -1)]
            cond_batch += [cond.T]
            if f0 is not None:
                f0_batch += [f0.reshape(1, -1)]

        # Convert lists to tensors, assuming each item has the same length
        audio_batch = torch.FloatTensor(
            np.array(audio_batch)
        )  # (batch, 1, frames * hop_length)
        cond_batch = torch.FloatTensor(np.array(cond_batch))  # (batch, dim, frames)
        f0_batch = (
            torch.FloatTensor(np.array(f0_batch)) if f0 is not None else None
        )  # (batch, 1, frames)

        return audio_batch, cond_batch, f0_batch

    def _check_length(self, audio: ndarray, cond: ndarray, f0: ndarray) -> None:
        """Assert the audio and feature lengths are correctly adjusted for upsampling."""
        assert len(audio) == len(cond) * self.hop_length
        if f0 is not None:
            assert len(cond) == len(f0)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    """
    Run the training process based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration parameters for training.
    """
    # Set the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"{'GPU' if torch.cuda.is_available() else 'CPU'} detected.")

    # Enable cuDNN benchmark for improved performance when using fixed size inputs
    # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Fix random seed for reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)

    # Create output directory if it does not exist
    if not os.path.exists(cfg.out_dir):
        os.makedirs(cfg.out_dir)

    # Write configuration to a YAML file
    with open(os.path.join(cfg.out_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(OmegaConf.to_yaml(cfg))

    # Load scaler for normalizing features
    scaler = load(to_absolute_path(cfg.data.stats))

    # Define models and optimizers
    model = {
        "generator": instantiate(cfg.generator).to(device),
        "discriminator": instantiate(cfg.discriminator).to(device),
    }

    # Handle distributed training if enabled
    if cfg.train.distributed_training:
        model["generator"] = torch.nn.DataParallel(model["generator"])
        model["discriminator"] = torch.nn.DataParallel(model["discriminator"])

    # Define optimizers and schedulers
    optimizer = {
        "generator": instantiate(
            cfg.train.generator_optimizer,
            params=model["generator"].parameters(),
        ),
        "discriminator": instantiate(
            cfg.train.discriminator_optimizer,
            params=model["discriminator"].parameters(),
        ),
    }
    scheduler = {
        "generator": instantiate(
            cfg.train.generator_scheduler, optimizer=optimizer["generator"]
        ),
        "discriminator": instantiate(
            cfg.train.discriminator_scheduler, optimizer=optimizer["discriminator"]
        ),
    }

    # Define training criteria
    criterion = {
        "mel": instantiate(cfg.train.mel_loss).to(device),
        "adv": instantiate(cfg.train.adv_loss).to(device),
        "fm": instantiate(cfg.train.fm_loss).to(device),
    }
    if cfg.train.lambda_reg > 0:
        criterion["reg"] = instantiate(cfg.train.reg_loss).to(device)

    # Prepare training and validation datasets
    sample_rate = cfg.generator.sample_rate
    if hasattr(cfg.generator, "hop_length"):
        hop_length = cfg.generator.hop_length
    else:
        hop_length = np.prod(cfg.generator.upsample_scales)
    feat_length_threshold = None
    if cfg.data.remove_short_samples:
        feat_length_threshold = cfg.data.batch_max_length // hop_length

    train_dataset = AudioFeatDataset(
        scaler=scaler,
        audio_list=to_absolute_path(cfg.data.train_audio),
        feat_list=to_absolute_path(cfg.data.train_feat),
        feat_length_threshold=feat_length_threshold,
        feat_names=cfg.data.feat_names,
        use_continuous_f0=cfg.data.use_continuous_f0,
        sample_rate=sample_rate,
        hop_length=hop_length,
        allow_cache=cfg.data.allow_cache,
    )
    logger.info(f"The number of training files = {len(train_dataset)}.")

    valid_dataset = AudioFeatDataset(
        scaler=scaler,
        audio_list=to_absolute_path(cfg.data.valid_audio),
        feat_list=to_absolute_path(cfg.data.valid_feat),
        feat_length_threshold=feat_length_threshold,
        feat_names=cfg.data.feat_names,
        use_continuous_f0=cfg.data.use_continuous_f0,
        sample_rate=sample_rate,
        hop_length=hop_length,
        allow_cache=cfg.data.allow_cache,
    )
    logger.info(f"The number of validation files = {len(valid_dataset)}.")

    dataset = {"train": train_dataset, "valid": valid_dataset}

    # Prepare data loader
    collater = Collater(
        batch_max_length=cfg.data.batch_max_length,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    train_sampler, valid_sampler = None, None
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=True,
            collate_fn=collater,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            sampler=train_sampler,
            pin_memory=cfg.data.pin_memory,
        ),
        "valid": DataLoader(
            dataset=dataset["valid"],
            shuffle=True,
            collate_fn=collater,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            sampler=valid_sampler,
            pin_memory=cfg.data.pin_memory,
        ),
    }

    # Setup trainer
    trainer = Trainer(
        cfg=cfg,
        steps=0,
        epochs=0,
        data_loader=data_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # Load trained parameters from checkpoint
    if cfg.train.resume:
        resume = os.path.join(
            cfg.out_dir, "checkpoints", f"checkpoint-{cfg.train.resume}steps.pkl"
        )
        if os.path.exists(resume):
            trainer.load_checkpoint(resume, cfg.train.load_only_params)
            logger.info(f"Successfully resumed from {resume}.")
        else:
            logger.info(f"Failed to resume from {resume}.")
            sys.exit(0)
    else:
        logger.info("Start a new training process.")

    # Run the training process
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(
            os.path.join(
                cfg.out_dir, "checkpoints", f"checkpoint-{trainer.steps}steps.pkl"
            )
        )
        logger.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
