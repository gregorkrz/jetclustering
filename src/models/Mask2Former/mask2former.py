"""
Mask2Former-style jet segmentation model.

Each event's PF candidates are encoded by a transformer backbone, then M learnable
queries cross-attend to the particle features over L decoder layers.  Each query
predicts (a) a binary class (jet vs. no-object) and (b) a soft mask over particles.
At inference the argmax of the mask logits gives a direct per-particle cluster
assignment without any HDBSCAN clustering step.

Loss: Hungarian matching between the M queries and the GT jets (one per dark quark
inside gt_radius), followed by BCE + Dice mask losses on matched pairs and a
cross-entropy class loss on all queries.

IRC safety loss (optional): for each particle that reappears in the augmented event
the soft query-assignment distribution should be invariant.  Query correspondence
across the two events is resolved by a single Hungarian step on mask cosine
similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from xformers.ops.fmha import BlockDiagonalMask

from src.models.transformer.tr_blocks import Transformer


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean Dice loss over the first dimension.

    inputs : [K, N]  probability scores (after sigmoid)
    targets: [K, N]  binary {0, 1}
    """
    num = 2 * (inputs * targets).sum(dim=-1)
    den = inputs.sum(dim=-1) + targets.sum(dim=-1)
    return (1 - (num + 1) / (den + 1)).mean()


def _batch_bce_cost(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE cost matrix for Hungarian.

    probs  : [M, N]  predicted probabilities
    targets: [K, N]  binary GT masks
    Returns: [M, K]
    """
    pos = F.binary_cross_entropy(probs, torch.ones_like(probs), reduction="none")   # [M, N]
    neg = F.binary_cross_entropy(probs, torch.zeros_like(probs), reduction="none")  # [M, N]
    cost = torch.einsum("mn,kn->mk", pos, targets) + torch.einsum("mn,kn->mk", neg, 1 - targets)
    return cost / targets.shape[-1]


def _batch_dice_cost(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Dice cost matrix for Hungarian.

    probs  : [M, N]
    targets: [K, N]
    Returns: [M, K]
    """
    num = 2 * torch.einsum("mn,kn->mk", probs, targets)
    den = probs.sum(dim=-1)[:, None] + targets.sum(dim=-1)[None, :]
    return 1 - (num + 1) / (den + 1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Mask2FormerDecoderLayer(nn.Module):
    """One Mask2Former decoder layer (pre-LN): self-attn → masked cross-attn → FFN."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1      = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2      = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm3      = nn.LayerNorm(d_model)
        self.mask_embed = nn.Linear(d_model, d_model)

    def forward(
        self,
        queries: torch.Tensor,                    # [M, D]
        pixel_features: torch.Tensor,             # [N_evt, D]
        prev_mask_logits: torch.Tensor | None,    # [M, N_evt] or None
    ):
        # --- self-attention ---
        q = self.norm1(queries).unsqueeze(0)           # [1, M, D]
        q, _ = self.self_attn(q, q, q)
        queries = queries + q.squeeze(0)

        # --- masked cross-attention ---
        attn_mask = None
        if prev_mask_logits is not None:
            # True = ignore; stop gradient through attention mask
            attn_mask = (prev_mask_logits.detach().sigmoid() < 0.5)   # [M, N_evt]
            # Guarantee each query can attend to at least one particle
            all_masked = attn_mask.all(dim=1, keepdim=True)
            attn_mask = attn_mask & ~all_masked

        q = self.norm2(queries).unsqueeze(0)           # [1, M, D]
        kv = pixel_features.unsqueeze(0)               # [1, N_evt, D]
        q, _ = self.cross_attn(q, kv, kv, attn_mask=attn_mask)
        queries = queries + q.squeeze(0)

        # --- FFN ---
        queries = queries + self.ffn(self.norm3(queries))

        # --- intermediate mask prediction ---
        mask_logits = torch.einsum(
            "md,nd->mn", self.mask_embed(queries), pixel_features
        )  # [M, N_evt]
        return queries, mask_logits


class Mask2FormerModel(nn.Module):
    def __init__(
        self,
        n_scalars: int,
        n_enc_blocks: int,
        n_dec_layers: int,
        n_heads: int,
        d_model: int,
        n_queries: int,
    ) -> None:
        super().__init__()
        self.n_queries = n_queries
        n_in = n_scalars + 3  # scalar features + (px, py, pz)
        self.bn         = nn.BatchNorm1d(n_in)
        self.input_proj = nn.Linear(n_in, d_model)
        self.encoder = Transformer(
            in_channels=d_model,
            out_channels=d_model,
            hidden_channels=d_model,
            num_blocks=n_enc_blocks,
            num_heads=n_heads,
        )
        self.query_embed = nn.Embedding(n_queries, d_model)
        self.decoder = nn.ModuleList([
            Mask2FormerDecoderLayer(d_model, n_heads)
            for _ in range(n_dec_layers)
        ])
        self.class_head = nn.Linear(d_model, 2)   # 0 = jet, 1 = no-object

    def forward(self, data):
        feats = torch.cat(
            [data.input_scalars.float(), data.input_vectors.float()], dim=1
        )
        feats = self.bn(feats)
        x = self.input_proj(feats)   # [N_total, D]

        # Backbone encoder with per-event block-diagonal attention
        seqlens  = torch.bincount(data.batch_idx.long()).tolist()
        enc_mask = BlockDiagonalMask.from_seqlens(seqlens)
        pixel_features = self.encoder(x.unsqueeze(0), attention_mask=enc_mask)[0]  # [N_total, D]

        # Per-event decoder loop
        n_events = int(data.batch_idx.max().item()) + 1
        class_logits_list, mask_logits_list = [], []
        for evt_i in range(n_events):
            evt_sel    = (data.batch_idx == evt_i)
            evt_pixels = pixel_features[evt_sel]          # [N_evt, D]
            queries    = self.query_embed.weight.clone()  # [M, D]
            prev_mask  = None
            for layer in self.decoder:
                queries, prev_mask = layer(queries, evt_pixels, prev_mask)
            class_logits_list.append(self.class_head(queries))  # [M, 2]
            mask_logits_list.append(prev_mask)                  # [M, N_evt]

        return {"class_logits": class_logits_list, "mask_logits": mask_logits_list}


# ---------------------------------------------------------------------------
# Cluster label extraction
# ---------------------------------------------------------------------------

def predictions_to_cluster_labels(
    y_pred: dict, batch_idx: torch.Tensor
) -> torch.Tensor:
    """Convert Mask2Former output to per-particle integer cluster labels.

    Returns a CPU tensor of shape [N_total] with values in {-1, 0, 1, ...} where
    -1 means noise (particle assigned to a no-object query).
    """
    class_logits_list = y_pred["class_logits"]
    mask_logits_list  = y_pred["mask_logits"]
    n_events = len(class_logits_list)
    all_labels = torch.full((len(batch_idx),), -1, dtype=torch.long)

    for evt_i in range(n_events):
        evt_sel      = (batch_idx == evt_i)
        class_logits = class_logits_list[evt_i].cpu()   # [M, 2]
        mask_logits  = mask_logits_list[evt_i].cpu()    # [M, N_evt]

        is_jet          = class_logits.argmax(dim=1) == 0   # [M]
        particle_query  = mask_logits.argmax(dim=0)          # [N_evt]
        particle_labels = torch.where(
            is_jet[particle_query], particle_query, torch.full_like(particle_query, -1)
        )

        # Renumber so jet-query indices become 0, 1, 2, ...
        jet_indices = is_jet.nonzero(as_tuple=True)[0]
        if len(jet_indices) > 0:
            remap = torch.full((class_logits.shape[0],), -1, dtype=torch.long)
            remap[jet_indices] = torch.arange(len(jet_indices), dtype=torch.long)
            particle_labels = torch.where(
                particle_labels >= 0, remap[particle_labels], particle_labels
            )

        all_labels[evt_sel] = particle_labels

    return all_labels


# ---------------------------------------------------------------------------
# Training loss (Hungarian matching)
# ---------------------------------------------------------------------------

def _mask2former_loss(
    batch,
    y_pred: dict,
    gt_labels: torch.Tensor,
    ce_weight: float = 2.0,
    bce_weight: float = 5.0,
    dice_weight: float = 5.0,
    no_object_coeff: float = 0.1,
) -> tuple:
    """Per-batch Mask2Former loss with Hungarian matching."""
    class_logits_list = y_pred["class_logits"]
    mask_logits_list  = y_pred["mask_logits"]
    n_events = len(class_logits_list)
    dev = class_logits_list[0].device

    total_loss = torch.zeros(1, device=dev).squeeze()
    sum_bce, sum_dice, sum_ce = 0.0, 0.0, 0.0

    for evt_i in range(n_events):
        evt_sel      = (batch.batch_idx == evt_i)
        evt_gt       = gt_labels[evt_sel]              # {-1, 0, 1, ...}
        class_logits = class_logits_list[evt_i]        # [M, 2]
        mask_logits  = mask_logits_list[evt_i]         # [M, N_evt]
        M = class_logits.shape[0]

        unique_jets = evt_gt[evt_gt >= 0].unique()
        n_gt        = len(unique_jets)
        ce_targets  = torch.ones(M, dtype=torch.long, device=dev)  # all → no-object

        if n_gt > 0:
            gt_masks = torch.stack(
                [(evt_gt == j).float() for j in unique_jets], dim=0
            ).to(dev)  # [n_gt, N_evt]

            with torch.no_grad():
                probs      = mask_logits.sigmoid()   # [M, N_evt]
                cost       = (
                    ce_weight   * (-F.softmax(class_logits, dim=-1)[:, 0:1].expand(-1, n_gt))
                    + bce_weight  * _batch_bce_cost(probs, gt_masks)
                    + dice_weight * _batch_dice_cost(probs, gt_masks)
                )
                rows, cols = linear_sum_assignment(cost.cpu().float().numpy())

            matched_pred = mask_logits[rows]   # [n_gt, N_evt]
            matched_gt   = gt_masks[cols]      # [n_gt, N_evt]
            loss_bce  = F.binary_cross_entropy_with_logits(matched_pred, matched_gt)
            loss_dice = _dice_loss(matched_pred.sigmoid(), matched_gt)
            ce_targets[torch.tensor(rows, device=dev)] = 0   # matched → jet
            sum_bce  += loss_bce.item()
            sum_dice += loss_dice.item()
        else:
            loss_bce  = torch.zeros(1, device=dev).squeeze()
            loss_dice = torch.zeros(1, device=dev).squeeze()

        ce_class_weight = torch.tensor([1.0, no_object_coeff], device=dev)
        loss_ce = F.cross_entropy(class_logits, ce_targets, weight=ce_class_weight)
        sum_ce += loss_ce.item()

        evt_loss = ce_weight * loss_ce + bce_weight * loss_bce + dice_weight * loss_dice
        total_loss = total_loss + evt_loss

    total_loss = total_loss / n_events
    loss_dict = {
        "loss_m2f_ce":   torch.tensor(sum_ce   / n_events),
        "loss_m2f_bce":  torch.tensor(sum_bce  / n_events),
        "loss_m2f_dice": torch.tensor(sum_dice / n_events),
    }
    return total_loss, loss_dict


def get_loss_func(args):
    def loss(model_input, model_output, gt_labels):
        return _mask2former_loss(model_input, model_output, gt_labels)
    return loss


# ---------------------------------------------------------------------------
# IRC safety loss
# ---------------------------------------------------------------------------

def _apply_batch_offsets(original_particle_mapping, event, event_aug):
    """Add per-event offsets to original_particle_mapping (mirrors loss_func_aug)."""
    to_add      = event.pfcands.batch_number[:-1]
    aug_batch   = event_aug.pfcands.batch_number
    filt_idx    = torch.where(original_particle_mapping != -1)[0].tolist()
    for i in range(len(aug_batch) - 1):
        for item in filt_idx:
            if aug_batch[i] <= item < aug_batch[i + 1]:
                original_particle_mapping[item] += to_add[i]
    return original_particle_mapping


def get_irc_loss_func():
    """Return the IRC safety loss function for Mask2Former.

    For each original particle reappearing in the augmented event, the soft
    query-assignment distributions (softmax of mask logits over queries) should be
    similar.  Query correspondence between the two events is found by Hungarian
    matching on mask cosine similarity, so the loss is permutation-equivariant.
    """
    def irc_loss(y_pred, y_pred_aug, batch, batch_aug, event, event_aug):
        opm = _apply_batch_offsets(
            batch_aug.original_particle_mapping.clone().cpu(), event, event_aug
        )

        n_events = len(y_pred["mask_logits"])
        dev      = y_pred["mask_logits"][0].device
        total    = torch.zeros(1, device=dev).squeeze()
        n_valid  = 0

        for evt_i in range(n_events):
            orig_sel = (batch.batch_idx     == evt_i)
            aug_sel  = (batch_aug.batch_idx == evt_i)
            mask_orig = y_pred["mask_logits"][evt_i]      # [M, N_orig]
            mask_aug  = y_pred_aug["mask_logits"][evt_i]  # [M, N_aug]
            M = mask_orig.shape[0]

            opm_evt = opm[aug_sel.cpu()]   # [N_aug], global orig indices (or -1)
            aug_local_mapped = torch.where(opm_evt != -1)[0]   # local aug indices
            if len(aug_local_mapped) == 0:
                continue

            orig_start  = orig_sel.nonzero()[0].item()
            orig_global = opm_evt[aug_local_mapped]             # global orig indices
            orig_local  = orig_global - orig_start              # local orig indices
            if orig_local.max() >= mask_orig.shape[1]:
                continue

            orig_local_dev   = orig_local.to(dev)
            aug_local_dev    = aug_local_mapped.to(dev)

            # Soft assignment distributions for the mapped particles
            soft_orig = F.softmax(mask_orig[:, orig_local_dev], dim=0).detach()  # [M, K]
            soft_aug  = F.softmax(mask_aug[:, aug_local_dev],   dim=0)           # [M, K]

            # Align queries via Hungarian on mask cosine similarity [M, M]
            with torch.no_grad():
                sim = (
                    F.normalize(mask_orig, dim=1)
                    @ F.normalize(mask_aug, dim=1).T
                )
                rows, cols = linear_sum_assignment(-sim.cpu().float().numpy())

            perm = torch.zeros(M, dtype=torch.long, device=dev)
            perm[torch.tensor(rows, device=dev)] = torch.tensor(cols, device=dev)
            soft_aug_aligned = soft_aug[perm]   # [M, K], reordered to match orig

            total   = total + F.mse_loss(soft_aug_aligned, soft_orig)
            n_valid += 1

        return total / max(n_valid, 1)

    return irc_loss


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(args, obj_score=False):
    n_scalars = 12
    if args.no_pid:
        n_scalars = 3
    return Mask2FormerModel(
        n_scalars=n_scalars,
        n_enc_blocks=args.num_blocks,
        n_dec_layers=args.num_dec_layers,
        n_heads=args.n_heads,
        d_model=args.internal_dim,
        n_queries=args.num_queries,
    )
