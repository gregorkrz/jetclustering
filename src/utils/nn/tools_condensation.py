import numpy as np
import awkward as ak
import tqdm
import time
import torch
from collections import defaultdict, Counter

from src.utils.metrics import evaluate_metrics
from src.data.tools import _concat
from src.logger.logger import _logger
from torch_scatter import scatter_sum, scatter_max
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pathlib import Path
from src.layers.object_cond import calc_eta_phi
import os
import pickle
from src.dataset.functions_data import get_batch, get_corrected_batch
from src.plotting.plot_event import plot_batch_eval_OC, get_labels_jets
from src.jetfinder.clustering import get_clustering_labels
from src.evaluation.clustering_metrics import compute_f1_score_from_result
from src.utils.train_utils import get_target_obj_score, plot_obj_score_debug # for debugging only!


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
    batch_config=None,
    val_dataset=None,
    obj_score_model=None,
    opt_obj_score=None,
    sched_obj_score=None
):
    if obj_score_model is None:
        model.train()
    else:
        obj_score_model.train()
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
        if obj_score_model is not None:
            opt_obj_score.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
            batch.to(dev)
        model_forward_time_start = time.time()
        if obj_score_model is not None:
            with torch.no_grad():
                y_pred = model(batch) # Only train the objectness score model
        else:
            y_pred = model(batch)
        model_forward_time_end = time.time()
        loss, loss_dict = loss_func(batch, y_pred, y)
        loss_time_end = time.time()
        wandb.log({
            "time_preprocess": time_preprocess_end - time_preprocess_start,
            "time_model_forward": model_forward_time_end - model_forward_time_start,
            "time_loss": loss_time_end - model_forward_time_end,
        }, step=step_count)
        if obj_score_model is not None:
            # Compute the objectness score
            coords = y_pred[:, 1:4]
            # TODO: update this to match the model architecture, as it's written here it's only suitable for L-GATr
            _, clusters, event_idx_clusters = get_clustering_labels(coords.detach().cpu().numpy(),
                                             batch.batch_idx.detach().cpu().numpy(),
                                             min_cluster_size=args.min_cluster_size,
                                             min_samples=args.min_samples, epsilon=args.epsilon,
                                             return_labels_event_idx=True)
           # Loop through the events in a batch
            input_pxyz = event_batch.pfcands.pxyz[batch.filter.cpu()]
            #input_pt = torch.sqrt(torch.sum(input_pxyz[:, :2] ** 2, dim=-1))
            clusters_pxyz = scatter_sum(input_pxyz, torch.tensor(clusters) + 1, dim=0)[1:]
            #clusters_highest_pt_particle = scatter_max(input_pt, torch.tensor(clusters) + 1, dim=0)[0][1:]
            clusters_eta, clusters_phi = calc_eta_phi(clusters_pxyz, return_stacked=False)
            #pfcands_eta, pfcands_phi = calc_eta_phi(input_pxyz, return_stacked=False)
            clusters_pt = torch.norm(clusters_pxyz[:, :2], dim=-1)
            filter = clusters_pt >= 100  # Don't train on the clusters that eventually get cut off
            batch_corr = get_corrected_batch(batch, clusters)
            if not args.global_features_obj_score:
                objectness_score = obj_score_model(batch_corr)[filter].flatten() # Obj. score is [0, 1]
            else:
                objectness_score = obj_score_model(batch_corr, batch, clusters)[filter].flatten()
            target_obj_score = get_target_obj_score(clusters_eta[filter], clusters_phi[filter], clusters_pt[filter],
                                                    torch.tensor(event_idx_clusters)[filter], y.dq_eta, y.dq_phi,
                                                    y.dq_coords_batch_idx, gt_mode=args.objectness_score_gt_mode)
            #target_obj_score = clusters_highest_pt_particle[filter].to(objectness_score.device)
            #fig = plot_obj_score_debug(y.dq_eta, y.dq_phi, y.dq_coords_batch_idx, clusters_eta[filter], clusters_phi[filter], clusters_pt[filter],
            #                           torch.tensor(event_idx_clusters)[filter], target_obj_score, input_pxyz, batch.batch_idx.detach().cpu(), torch.tensor(clusters), objectness_score)
            #fig.savefig(os.path.join(args.run_path, "obj_score_debug_{}.pdf".format(step_count)))
            n_positive, n_negative = target_obj_score.sum(), (1-target_obj_score).sum()
            # set weights for the loss according to the class imbalance
            #pos_weight = n_negative / (n_positive + n_negative)
            #neg_weight = n_positive / (n_positive + n_negative)
            n_all = n_positive + n_negative
            pos_weight = n_all / n_positive if n_positive > 0 else 0
            neg_weight = n_all / n_negative if n_negative > 0 else 0
            #print("Positive weight:", pos_weight, "Negative weight:", neg_weight)
            #weight = pos_weight * target_obj_score + neg_weight * (1 - target_obj_score)
            # Weights for BCELoss: per-element weight
            weights = torch.where(target_obj_score == 1, pos_weight, neg_weight)
            print("N positive:", n_positive.item(), "N negative:", n_negative.item())
            print("First 20 predictions:", objectness_score[:20], "First 20 targets:", target_obj_score[:20])
            objectness_score = objectness_score.clamp(min=-10, max=10)
            target_obj_score = target_obj_score.to(objectness_score.device)
            weights = weights.to(objectness_score.device)
            ##### TEMPORARY: PREDICT HIGHEST PT OF PARTICLE !!!!!! ######
            #loss_obj_score = torch.mean(torch.square(target_obj_score - objectness_score)) # temporarily just regress the highest pt particle to check for expresiveness of the model
            loss_obj_score = torch.nn.BCEWithLogitsLoss(weight=weights)(objectness_score, target_obj_score)
            #loss_obj_score = torch.mean(weights * (objectness_score - target_obj_score) ** 2)
            loss = loss_obj_score
            loss_dict["loss_obj_score"] = loss_obj_score
        if obj_score_model is None:
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()
        else:
            if grad_scaler is None:
                loss.backward()
                opt_obj_score.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt_obj_score)
                grad_scaler.update()
        step_end_time = time.time()
        loss = loss.item()
        wandb.log({key: value.detach().cpu().item() for key, value in loss_dict.items()}, step=step_count)
        wandb.log({"loss": loss}, step=step_count)
        del loss_dict
        del loss
        if (local_rank == 0) and (step_count % args.validation_steps) == 0:
            dirname = args.run_path
            if obj_score_model is None:
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
            else:
                model_state_dict = (
                    obj_score_model.module.state_dict()
                    if isinstance(
                        model,
                        (
                            torch.nn.DataParallel,
                            torch.nn.parallel.DistributedDataParallel,
                        ),
                    )
                    else obj_score_model.state_dict()
                )
                sched_sd = {}
                if sched_obj_score is not None:
                    sched_sd = sched_obj_score.state_dict()
                state_dict = {"model": model_state_dict, "optimizer": opt_obj_score.state_dict(),
                              "scheduler": sched_sd}
                path = os.path.join(dirname, "OS_step_%d_epoch_%d.ckpt" % (step_count, epoch))
                torch.save(
                    state_dict,
                    path
                )
            res = evaluate(
                model,
                val_loader,
                dev,
                epoch,
                step_count,
                loss_func=loss_func,
                gt_func=gt_func,
                local_rank=local_rank,
                args=args,
                batch_config=batch_config,
                predict=False,
                model_obj_score=obj_score_model
            )
            if obj_score_model is not None:
                res, res_obj_score, res_obj_score1 = res
                # TODO: use the obj score here for quick evaluation
            f1 = compute_f1_score_from_result(res, val_dataset)
            wandb.log({"val_f1_score": f1}, step=step_count)
        if args.num_steps != -1 and step_count >= args.num_steps:
            print("Quitting training as the required number of steps has been reached.")
            return "quit_training"
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
    batch_config=None,
    predict=False,
    model_obj_score=None # if not None, it will compute the objectness score of each cluster using the proposed method
):
    model.eval()
    count = 0
    start_time = time.time()
    total_loss = 0
    total_loss_dict = {}
    plot_batches = [0, 1]
    n_batches = 0
    if predict or True: # predict also on validation set
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
            "radius_cluster_FatJets": [],
            "model_cluster": [],
            #"event_clusters_idx": []
        }
        if model_obj_score is not None:
            obj_score_predictions = []
            obj_score_targets = []
            predictions["event_clusters_idx"] = []
        if args.beta_type != "pt+bc":
            del predictions["BC_score"]
    last_event_idx = 0
    with torch.no_grad():
        with tqdm.tqdm(eval_loader) as tq:
            for event_batch in tq:
                count += event_batch.n_events # number of samples
                y = gt_func(event_batch)
                batch, y = get_batch(event_batch, batch_config, y, test=predict)
                y = y.to(dev)
                batch = batch.to(dev)
                y_pred = model(batch)
                if not predict:
                    loss, loss_dict = loss_func(batch, y_pred, y)
                    loss = loss.item()
                    total_loss += loss
                    for key in loss_dict:
                        if key not in total_loss_dict:
                            total_loss_dict[key] = 0
                        total_loss_dict[key] += loss_dict[key].item()
                    del loss_dict
                if n_batches in plot_batches and not predict: # don't plot these for prediction - they are useful in training
                    plot_folder = os.path.join(args.run_path, "eval_plots", "epoch_" + str(epoch) + "_step_" + str(step))
                    Path(plot_folder).mkdir(parents=True, exist_ok=True)
                    if args.loss == "quark_distance":
                        label_true = y.labels_no_renumber.detach().cpu()
                    elif args.train_objectness_score:
                        label_true = y.labels.detach().cpu()
                    else:
                       label_true = y.detach().cpu()
                    #plot_batch_eval_OC(event_batch, label_true,
                    #                   y_pred.detach().cpu(), batch.batch_idx.detach().cpu(),
                    #                   os.path.join(plot_folder, "batch_" + str(n_batches) + ".pdf"),
                    #                   args=args, batch=n_batches, dropped_batches=batch.dropped_batches)
                n_batches += 1
                if not predict:
                    tq.set_postfix(
                        {
                            "Loss": "%.5f" % loss,
                            "AvgLoss": "%.5f" % (total_loss / n_batches),
                        }
                    )
                if predict or True:
                    event_idx = batch.batch_idx + last_event_idx
                    predictions["event_idx"].append(event_idx)
                    if not model_obj_score:
                        predictions["GT_cluster"].append(y.detach().cpu())
                    else:
                        predictions["GT_cluster"].append(y.labels.detach().cpu())
                    predictions["pred"].append(y_pred.detach().cpu())
                    predictions["eta"].append(event_batch.pfcands.eta.detach().cpu())
                    predictions["phi"].append(event_batch.pfcands.phi.detach().cpu())
                    predictions["pt"].append(event_batch.pfcands.pt.detach().cpu())
                    predictions["AK8_cluster"].append(event_batch.pfcands.pf_cand_jet_idx.detach().cpu())
                    predictions["radius_cluster_GenJets"].append(get_labels_jets(event_batch, event_batch.pfcands, event_batch.genjets).detach().cpu())
                    predictions["radius_cluster_FatJets"].append(get_labels_jets(event_batch, event_batch.pfcands, event_batch.fatjets).detach().cpu())
                    predictions["mass"].append(event_batch.pfcands.mass.detach().cpu())
                    if predictions["pred"][-1].shape[1] == 4:
                        coords = predictions["pred"][-1][:, :3]
                    else:
                        coords = predictions["pred"][-1][:, 1:4]
                    #if model_obj_score is None:
                    clustering_labels = torch.tensor(
                        get_clustering_labels(
                                coords.detach().cpu().numpy(),
                                event_idx.detach().cpu().numpy(),
                                min_cluster_size=args.min_cluster_size,
                                min_samples=args.min_samples,
                                epsilon=args.epsilon,
                                return_labels_event_idx=False)
                            )
                    if model_obj_score is not None:
                        _, clusters, event_idx_clusters = get_clustering_labels(coords.detach().cpu().numpy(),
                                                                             batch.batch_idx.detach().cpu().numpy(),
                                                                             min_cluster_size=args.min_cluster_size,
                                                                             min_samples=args.min_samples,
                                                                             epsilon=args.epsilon,
                                                                             return_labels_event_idx=True)
                        assert len(event_idx_clusters) == clusters.max() + 1
                        batch_corr = get_corrected_batch(batch, clusters)
                        input_pxyz = event_batch.pfcands.pxyz[batch.filter.cpu()]
                        clusters_pxyz = scatter_sum(input_pxyz, torch.tensor(clusters) + 1, dim=0)[1:]
                        clusters_eta, clusters_phi = calc_eta_phi(clusters_pxyz, return_stacked=False)
                        # pfcands_eta, pfcands_phi = calc_eta_phi(input_pxyz, return_stacked=False)
                        clusters_pt = torch.norm(clusters_pxyz[:, :2], dim=-1)
                        filter = clusters_pt >= 100  # Don't train on the clusters that eventually get cut off
                        if not args.global_features_obj_score:
                            objectness_score = model_obj_score(batch_corr)
                        else:
                            objectness_score = model_obj_score(batch_corr, batch, clusters)
                        obj_score_predictions.append(objectness_score.detach().cpu())
                        target_obj_score = get_target_obj_score(clusters_eta[filter], clusters_phi[filter],
                                                                clusters_pt[filter],
                                                                torch.tensor(event_idx_clusters)[filter], y.dq_eta,
                                                                y.dq_phi, y.dq_coords_batch_idx, gt_mode=args.objectness_score_gt_mode)  # [filter]
                        n_positive, n_negative = target_obj_score.sum(), (1 - target_obj_score.float()).sum()
                        # set weights for the loss according to the class imbalance
                        # pos_weight = n_negative / (n_positive + n_negative)
                        # neg_weight = n_positive / (n_positive + n_negative)
                        n_all = n_positive + n_negative
                        pos_weight = n_all / n_positive if n_positive > 0 else 0
                        neg_weight = n_all / n_negative if n_negative > 0 else 0

                        # Weights for BCELoss: per-element weight
                        weights = torch.where(target_obj_score == 1, pos_weight, neg_weight)
                        print("N positive (eval):", n_positive.item(), "N negative (eval):", n_negative.item())
                        print("First 10 predictions (eval):", objectness_score[:20], "First 10 targets (eval):",
                              target_obj_score[:20])
                        objectness_score = objectness_score.clamp(min=-10, max=10)
                        target_obj_score = target_obj_score.to(objectness_score.device)
                        #print(target_obj_score.device, filter.device, objectness_score.device, weights.device)
                        weights = weights.to(objectness_score.device)
                        filter = filter.to(objectness_score.device)
                        loss_obj_score = torch.nn.BCEWithLogitsLoss(weight=weights)(objectness_score.flatten()[filter], target_obj_score.flatten()).cpu().item()
                        # compute ROC AUC
                        obj_score_targets.append(target_obj_score)
                        k = "val_loss_obj_score"
                        if k not in total_loss_dict:
                            total_loss_dict[k] = 0
                        total_loss_dict[k] += loss_obj_score
                        predictions["event_clusters_idx"].append(torch.tensor(event_idx_clusters) + last_event_idx)
                        # loss_obj_score = torch.mean(weights * (objectness_score - target_obj_score) ** 2)
                    predictions["model_cluster"].append(
                        torch.tensor(clustering_labels)
                    )
                    last_event_idx = event_idx.max().item() + 1
    if local_rank == 0 and not predict:
        wandb.log({"val_loss": total_loss / n_batches}, step=step)
        wandb.log({"val_" + key: value / n_batches for key, value in total_loss_dict.items()}, step=step)

    time_diff = time.time() - start_time
    _logger.info(
        "Evaluated on %d samples in total (avg. speed %.1f samples/s)"
        % (count, count / time_diff)
    )
    if predict or True:
        #for key in predictions:
        #    predictions[key] = torch.cat(predictions[key], dim=0)
        #predictions = {key: torch.cat(predictions[key], dim=0) for key in predictions}
        predictions_1 = {}
        for key in predictions:
            print("key", key, predictions[key])
            predictions_1[key] = torch.cat(predictions[key], dim=0)
        predictions = predictions_1
        #predictions["event_idx"] = torch.cat(predictions["event_idx"], dim=0)
        #predictions["GT_cluster"] = torch.cat(predictions["GT_cluster"], dim=0)
        #predictions["pred"] = torch.cat(predictions["pred"], dim=0)
        #predictions["eta"] = torch.cat(predictions["eta"], dim=0)
        #predictions["phi"] = torch.cat(predictions["phi"], dim=0)
        #predictions["pt"] = torch.cat(predictions["pt"], dim=0)
        #predictions["AK8_cluster"] = torch.cat(predictions["AK8_cluster"], dim=0)
        #predictions["radius_cluster_GenJets"] = torch.cat(predictions["radius_cluster_GenJets"], dim=0)
        #predictions["radius_cluster_FatJets"] = torch.cat(predictions["radius_cluster_FatJets"], dim=0)
        #predictions["mass"] = torch.cat(predictions["mass"], dim=0)
        #predictions["model_cluster"] = torch.cat(predictions["model_cluster"], dim=0)
        if model_obj_score is not None:
            return predictions, torch.cat(obj_score_predictions), torch.cat(obj_score_targets)
        return predictions
    return total_loss / count # Average loss is the validation metric here
