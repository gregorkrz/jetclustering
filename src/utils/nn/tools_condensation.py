import numpy as np
import awkward as ak
import tqdm
import time
import torch
from collections import defaultdict, Counter
from src.utils.metrics import evaluate_metrics
from src.data.tools import _concat
from src.logger.logger import _logger
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pathlib import Path
import os
import pickle
from src.models.gravnet_calibration import object_condensation_loss2
from src.layers.inference_oc import create_and_store_graph_output
#from src.layers.object_cond import onehot_particles_arr
from src.utils.logger_wandb import plot_clust
from src.dataset.functions_data import get_batch
# class_names = ["other"] + [str(i) for i in onehot_particles_arr]  # quick fix
from src.plotting.plot_event import plot_batch_eval_OC

def train_epoch(
    args,
    model,
    loss_func,
    gt_func,
    opt,
    scheduler,
    train_loader,
    dev,
    epoch,
    grad_scaler=None,
    local_rank=0,
    current_step=0,
    val_loader=None
):
    model.train()
    step_count = current_step
    start_time = time.time()
    prev_time = time.time()
    for event_batch in tqdm.tqdm(train_loader):
        time_preprocess_start = time.time()
        y = gt_func(event_batch)
        batch = get_batch(event_batch, {})
        time_preprocess_end = time.time()
        step_count += 1
        y = y.to(dev)
        opt.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
            batch.to(dev)
        model_forward_time_start = time.time()
        y_pred = model(batch)
        model_forward_time_end = time.time()
        loss, loss_dict = loss_func(batch, y_pred, y)
        loss_time_end = time.time()
        wandb.log({
            "time_preprocess": time_preprocess_end - time_preprocess_start,
            "time_model_forward": model_forward_time_end - model_forward_time_start,
            "time_loss": loss_time_end - model_forward_time_end,
        }, step=step_count)
        if grad_scaler is None:
            loss.backward()
            opt.step()
        else:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(opt)
            grad_scaler.update()
        step_end_time = time.time()
        loss = loss.item()
        wandb.log({key: value.detach().cpu().item() for key, value in loss_dict.items()}, step=step_count)
        wandb.log({"loss": loss}, step=step_count)
        del loss_dict
        del loss
        if (local_rank == 0) and (step_count % 500) == 0:
            dirname = args.run_path
            model_state_dict = (
                model.module.state_dict()
                if isinstance(
                    model,
                    (
                        torch.nn.DataParallel,
                        torch.nn.parallel.DistributedDataParallel,
                    ),
                )
                else model.state_dict()
            )
            state_dict = {"model": model_state_dict, "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict()}
            path = os.path.join(dirname, "step_%d_epoch_%d.ckpt" % (step_count, epoch))
            torch.save(
                state_dict,
                path
            )
            evaluate(
                model,
                val_loader,
                dev,
                epoch,
                step_count,
                loss_func=loss_func,
                gt_func=gt_func,
                local_rank=local_rank,
                args=args,
            )
        #_logger.info(
        #    "Epoch %d, step %d: loss=%.5f, time=%.2fs"
        #    % (epoch, step_count, loss, step_end_time - prev_time)
        #)
    time_diff = time.time() - start_time
    return step_count


def evaluate(
    model,
    eval_loader,
    dev,
    epoch,
    step,
    loss_func,
    gt_func,
    local_rank=0,
    args=None,
):
    model.eval()
    count = 0
    start_time = time.time()
    total_loss = 0
    total_loss_dict = {}
    plot_batches = [0, 1]
    n_batches = 0
    with torch.no_grad():
        with tqdm.tqdm(eval_loader) as tq:
            for event_batch in tq:
                count += event_batch.n_events # number of samples
                y = gt_func(event_batch)
                batch = get_batch(event_batch, {})
                y = y.to(dev)
                batch = batch.to(dev)
                y_pred = model(batch)
                loss, loss_dict = loss_func(batch, y_pred, y)
                loss = loss.item()
                total_loss += loss
                for key in loss_dict:
                    if key not in total_loss_dict:
                        total_loss_dict[key] = 0
                    total_loss_dict[key] += loss_dict[key].item()
                if n_batches in plot_batches:
                    plot_folder = os.path.join(args.run_path, "eval_plots", "epoch_" + str(epoch) + "_step_" + str(step))
                    Path(plot_folder).mkdir(parents=True, exist_ok=True)
                    plot_batch_eval_OC(event_batch, y.detach().cpu(),
                                       y_pred.detach().cpu(), batch.batch_idx.detach().cpu(),
                                       os.path.join(plot_folder, "batch_" + str(n_batches) + ".pdf"), args=args)
                n_batches += 1
                tq.set_postfix(
                    {
                        "Loss": "%.5f" % loss,
                        "AvgLoss": "%.5f" % (total_loss / n_batches),
                    }
                )
                if args.predict:
                    pass # TODO: save the results here or do something with them
    if local_rank == 0:
        wandb.log({"val_loss": total_loss / n_batches}, step=step)
        wandb.log({"val_" + key: value / n_batches for key, value in total_loss_dict.items()}, step=step)

    time_diff = time.time() - start_time
    _logger.info(
        "Evaluated on %d samples in total (avg. speed %.1f samples/s)"
        % (count, count / time_diff)
    )
    return total_loss / count # average loss is the validation metric here
