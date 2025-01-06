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
from src.dataset.functions_data import get_batch
from src.plotting.plot_event import plot_batch_eval_OC, get_labels_jets

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
    val_loader=None,
    batch_config=None
):
    model.train()
    step_count = current_step
    start_time = time.time()
    prev_time = time.time()
    for event_batch in tqdm.tqdm(train_loader):
        time_preprocess_start = time.time()
        y = gt_func(event_batch)
        batch, y = get_batch(event_batch, batch_config, y)
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
        if (local_rank == 0) and (step_count % 1000) == 0:
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
                batch_config=batch_config
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
    batch_config=None
):
    model.eval()
    count = 0
    start_time = time.time()
    total_loss = 0
    total_loss_dict = {}
    plot_batches = [0, 1]
    n_batches = 0
    if args.predict:
        predictions = {
            "event_idx": [],
            "GT_cluster": [],
            "pred": [],
            "eta": [],
            "phi": [],
            "pt": [],
            "mass": [],
            "AK8_cluster": [],
            "radius_cluster_GenJets": [],
            "radius_cluster_FatJets": []
    }
        if args.beta_type != "pt+bc":
            del predictions["BC_score"]
    last_event_idx = 0
    with torch.no_grad():
        with tqdm.tqdm(eval_loader) as tq:
            for event_batch in tq:
                count += event_batch.n_events # number of samples
                y = gt_func(event_batch)
                batch, y = get_batch(event_batch, batch_config, y, test=args.predict)
                y = y.to(dev)
                batch = batch.to(dev)
                y_pred = model(batch)
                if not args.predict:
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
                                       os.path.join(plot_folder, "batch_" + str(n_batches) + ".pdf"), args=args, batch=n_batches)
                n_batches += 1
                if not args.predict:
                    tq.set_postfix(
                        {
                            "Loss": "%.5f" % loss,
                            "AvgLoss": "%.5f" % (total_loss / n_batches),
                        }
                    )
                if args.predict:
                    event_idx = batch.batch_idx + last_event_idx
                    predictions["event_idx"].append(event_idx)
                    predictions["GT_cluster"].append(y.detach().cpu())
                    predictions["pred"].append(y_pred.detach().cpu())
                    predictions["eta"].append(event_batch.pfcands.eta.detach().cpu())
                    predictions["phi"].append(event_batch.pfcands.phi.detach().cpu())
                    predictions["pt"].append(event_batch.pfcands.pt.detach().cpu())
                    predictions["AK8_cluster"].append(event_batch.pfcands.pf_cand_jet_idx.detach().cpu())
                    predictions["radius_cluster_GenJets"].append(get_labels_jets(event_batch, event_batch.pfcands, event_batch.genjets).detach().cpu())
                    predictions["radius_cluster_FatJets"].append(get_labels_jets(event_batch, event_batch.pfcands, event_batch.fatjets).detach().cpu())
                    predictions["mass"].append(event_batch.pfcands.mass.detach().cpu())
                    last_event_idx = event_idx.max().item() + 1
    if local_rank == 0 and not args.predict:
        wandb.log({"val_loss": total_loss / n_batches}, step=step)
        wandb.log({"val_" + key: value / n_batches for key, value in total_loss_dict.items()}, step=step)

    time_diff = time.time() - start_time
    _logger.info(
        "Evaluated on %d samples in total (avg. speed %.1f samples/s)"
        % (count, count / time_diff)
    )
    if args.predict:
        predictions["event_idx"] = torch.cat(predictions["event_idx"], dim=0)
        predictions["GT_cluster"] = torch.cat(predictions["GT_cluster"], dim=0)
        predictions["pred"] = torch.cat(predictions["pred"], dim=0)
        predictions["eta"] = torch.cat(predictions["eta"], dim=0)
        predictions["phi"] = torch.cat(predictions["phi"], dim=0)
        predictions["pt"] = torch.cat(predictions["pt"], dim=0)
        predictions["AK8_cluster"] = torch.cat(predictions["AK8_cluster"], dim=0)
        predictions["radius_cluster_GenJets"] = torch.cat(predictions["radius_cluster_GenJets"], dim=0)
        predictions["radius_cluster_FatJets"] = torch.cat(predictions["radius_cluster_FatJets"], dim=0)
        predictions["mass"] = torch.cat(predictions["mass"], dim=0)
        return predictions
    return total_loss / count # average loss is the validation metric here
