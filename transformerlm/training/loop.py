import os
from pathlib import Path
import torch
import numpy as np


def train_iterations(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    # data
    np_arr_train_data,
    np_arr_valid_data,
    # batching
    batch_size: int,
    context_length: int,
    device: str,
    # schedule
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    # iterations
    max_train_iteration: int | None,
    max_val_iteration: int | None,
    val_freq_iteration: int,
    # regularization
    grad_clip_max_l2_norm: float,
    # checkpointing
    ckpting_save_iter: int,
    ckpting_save_folder: Path | str | None,
    # helpers
    get_batch,
    cross_entropy,
    lr_cosine_schedule,
    gradient_clipping,
    save_checkpoint,
    # logging
    log=None,
    # optional logging helpers
    activation_norms: dict | None = None,
    log_activation_norms: bool = False,
    log_weight_norms: bool = False,
):
    """A minimal training loop extracted into a reusable function.

    The caller supplies batching, loss, schedule, clipping, and checkpoint helpers.
    If `log` is provided, it is called as `log(dict, step=iteration)`.
    """
    train_iteration = 0
    while True:
        model.train()
        train_batch_sampled_sequence, train_batch_sampled_ids = get_batch(
            dataset=np_arr_train_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # forward
        train_logits = model(train_batch_sampled_sequence)
        train_loss = cross_entropy(train_logits, train_batch_sampled_ids).mean()
        if log:
            log({"train_loss": float(train_loss.item())}, step=train_iteration)

        # activation norms (if hooks populate activation_norms)
        if log and log_activation_norms and activation_norms is not None and len(activation_norms) > 0:
            vals = list(activation_norms.values())
            log({
                "activation_norms/mean": float(np.mean(vals)),
                "activation_norms/max": float(np.max(vals)),
                "activation_norms/min": float(np.min(vals)),
                **{f"activation_norms/{k}": float(v) for k, v in activation_norms.items()},
            }, step=train_iteration)

        # backward
        optimizer.zero_grad()
        train_loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        l2_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]))
        if log:
            log({"grads": float(l2_norm.item())}, step=train_iteration)
        gradient_clipping(parameters=model.parameters(), max_l2_norm=grad_clip_max_l2_norm)
        optimizer.step()

        # weight norms
        if log and log_weight_norms:
            norms = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    norms[name] = float(param.data.norm().item())
            vals = list(norms.values())
            if vals:
                log({
                    "weight_norms/mean": float(np.mean(vals)),
                    "weight_norms/max": float(np.max(vals)),
                    "weight_norms/min": float(np.min(vals)),
                    **{f"weight_norms/{k}": v for k, v in norms.items()},
                }, step=train_iteration)

        # schedule
        new_lr = lr_cosine_schedule(train_iteration, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        if log:
            log({"lr": float(new_lr)}, step=train_iteration)

        # validation
        if train_iteration % val_freq_iteration == 0:
            model.eval()
            val_iteration = 0
            running_val_loss = 0.0
            while True:
                with torch.no_grad():
                    val_batch_sampled_sequence, val_batch_sampled_ids = get_batch(
                        dataset=np_arr_valid_data,
                        batch_size=batch_size,
                        context_length=context_length,
                        device=device,
                    )
                    val_logits = model(val_batch_sampled_sequence)
                    val_loss = cross_entropy(val_logits, val_batch_sampled_ids).mean()
                    running_val_loss += float(val_loss.item())
                val_iteration += 1
                if max_val_iteration is not None and val_iteration >= max_val_iteration:
                    break
            avg_val_loss = running_val_loss / (max_val_iteration if max_val_iteration else val_iteration)
            if log:
                log({"val_loss": float(avg_val_loss)}, step=train_iteration)

        # checkpoints
        if train_iteration > 0 and train_iteration % ckpting_save_iter == 0 and ckpting_save_folder is not None:
            ckpting_save_folder = Path(ckpting_save_folder)
            ckpting_save_folder.mkdir(parents=True, exist_ok=True)
            ckpt_file_iter = ckpting_save_folder / f"{train_iteration}.ckpt"
            save_checkpoint(model, optimizer, train_iteration, ckpt_file_iter)

        # termination
        if max_train_iteration is not None and train_iteration >= max_train_iteration:
            break
        train_iteration += 1
