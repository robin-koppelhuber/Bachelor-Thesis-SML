"""Multi-task model with T separate task-specific classification heads.

Wraps a RoBERTa backbone and provides T separate RobertaClassificationHead instances
(one per task) in an nn.ModuleDict.  The encoder (self.roberta) is shared; only the
active head changes per task.

Usage
-----
    model = create_multi_task_model(base_model_id, task_names, num_labels_per_task)
    model.set_task("cola")
    outputs = model(**batch)          # routes through heads["cola"]

Parameter naming convention
---------------------------
    roberta.*            →  shared backbone (encoder)
    heads.{task_name}.*  →  task-specific classification head

Setting base_model_prefix = "roberta" ensures that base.py's zeroing logic correctly
treats everything NOT starting with "roberta." as a classification head.  On a freshly
created MultiTaskModel the heads are randomly initialised (via RobertaClassificationHead's
own init), and the training base_state_dict zeros them — so the task-vector delta for
each head equals the full trained head weight (consistent with the zero-classifier base
used by the merging-method evaluation pipeline).
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

logger = logging.getLogger(__name__)


class MultiTaskModel(nn.Module):
    """RoBERTa backbone with T separate task-specific classification heads."""

    base_model_prefix = "roberta"

    def __init__(
        self,
        roberta_backbone: nn.Module,
        heads: nn.ModuleDict,
        config,
    ) -> None:
        """
        Args:
            roberta_backbone: Bare RobertaModel (not ForSequenceClassification).
            heads: nn.ModuleDict mapping task_name -> RobertaClassificationHead.
            config: RoBERTa config (kept for compatibility with HF utilities).
        """
        super().__init__()
        self.roberta = roberta_backbone
        self.heads = heads
        self.config = config
        self._current_task: Optional[str] = None

    # ------------------------------------------------------------------
    # Task routing
    # ------------------------------------------------------------------

    def set_task(self, task_name: str) -> None:
        """Set the active task head before the next forward pass."""
        if task_name not in self.heads:
            raise KeyError(
                f"Unknown task '{task_name}'. Available tasks: {list(self.heads.keys())}"
            )
        self._current_task = task_name

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        if self._current_task is None:
            raise RuntimeError(
                "Call set_task(task_name) before forward().  "
                f"Available tasks: {list(self.heads.keys())}"
            )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # RobertaClassificationHead extracts the [CLS] token (features[:, 0, :]) internally.
        sequence_output = outputs[0]
        logits = self.heads[self._current_task](sequence_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ------------------------------------------------------------------
    # HF compatibility helpers
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Delegate gradient checkpointing to the backbone."""
        if hasattr(self.roberta, "gradient_checkpointing_enable"):
            self.roberta.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


# ------------------------------------------------------------------
# Factory helpers
# ------------------------------------------------------------------

def _make_head_config(base_config, num_labels: int):
    """Return a copy of *base_config* with num_labels overridden."""
    from transformers import RobertaConfig
    d = base_config.to_dict()
    d["num_labels"] = num_labels
    return RobertaConfig(**d)


def create_multi_task_model(
    base_model_id: str,
    task_names: List[str],
    num_labels_per_task: Dict[str, int],
    cache_dir=None,
    device=None,
    torch_dtype: str = "auto",
) -> MultiTaskModel:
    """
    Create a MultiTaskModel from a pretrained base model.

    Heads are randomly initialised (standard RobertaClassificationHead init).
    The backbone uses pretrained weights from *base_model_id*.

    Args:
        base_model_id: HuggingFace model ID (e.g. ``"FacebookAI/roberta-base"``).
        task_names: Ordered list of task names.
        num_labels_per_task: Mapping task_name -> num_labels for each head.
        cache_dir: Optional local model cache directory.
        device: Optional device to place the model on.
        torch_dtype: Loading dtype (``"auto"``, ``"float32"``, ``"float16"``).

    Returns:
        :class:`MultiTaskModel` with shared backbone and T separate heads.
    """
    from src.models.loaders import load_model

    logger.info(f"Creating MultiTaskModel from {base_model_id}")
    logger.info(f"  Tasks         : {task_names}")
    logger.info(f"  Labels / task : {num_labels_per_task}")

    # Load a temporary ForSequenceClassification model to get the backbone
    tmp = load_model(base_model_id, num_labels=2, cache_dir=cache_dir, torch_dtype=torch_dtype)
    roberta_backbone = tmp.roberta
    base_config = tmp.config
    del tmp

    # Create one head per task (randomly initialised)
    heads = nn.ModuleDict({
        task: RobertaClassificationHead(_make_head_config(base_config, num_labels_per_task[task]))
        for task in task_names
    })

    model = MultiTaskModel(roberta_backbone, heads, base_config)

    if device is not None:
        model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    enc = sum(p.numel() for p in model.roberta.parameters())
    hds = sum(p.numel() for p in model.heads.parameters())
    logger.info(f"  Params: total={total:,}  encoder={enc:,}  heads={hds:,}")
    return model


# ------------------------------------------------------------------
# Task-vector helpers
# ------------------------------------------------------------------

def extract_per_task_state_dicts(
    multi_task_state: Dict[str, torch.Tensor],
    task_names: List[str],
    pretrained_encoder_state: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Convert a trained MultiTaskModel state dict into per-task state dicts
    in standard HuggingFace format (``roberta.*`` + ``classifier.*``).

    Mapping applied::

        roberta.*           →  roberta.*            (encoder delta = trained − pretrained)
        heads.{t}.dense.*   →  classifier.dense.*   (full weight; base was zeroed)
        heads.{t}.out_proj.*→  classifier.out_proj.*

    Args:
        multi_task_state: State dict of the trained :class:`MultiTaskModel`
            (keys like ``"roberta.*"`` and ``"heads.{t}.*"``).
        task_names: Task names (must match keys in *multi_task_state*).
        pretrained_encoder_state: State dict of the *pretrained* backbone
            (``roberta.*`` keys, CPU tensors) used to compute the encoder delta.

    Returns:
        ``{task_name: {param_name: tensor}}`` where each inner dict is a valid
        state dict for ``AutoModelForSequenceClassification`` with the matching
        ``num_labels``.
    """
    result: Dict[str, Dict[str, torch.Tensor]] = {}

    for task in task_names:
        task_state: Dict[str, torch.Tensor] = {}
        head_prefix = f"heads.{task}."

        for name, param in multi_task_state.items():
            if name.startswith("roberta."):
                # Encoder: store the delta (trained - pretrained)
                base_val = pretrained_encoder_state.get(name)
                if base_val is not None:
                    task_state[name] = param.cpu() - base_val.cpu()
                else:
                    task_state[name] = param.cpu()
            elif name.startswith(head_prefix):
                # Head: remap "heads.{task}.X" → "classifier.X"
                # Base head was zeroed → delta = full trained weight
                new_name = "classifier." + name[len(head_prefix):]
                task_state[new_name] = param.cpu()

        result[task] = task_state

    return result
