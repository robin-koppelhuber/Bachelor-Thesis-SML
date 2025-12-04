"""Base class for model merging methods"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ABC / abstract base class forces subclasses to implement all abstract methods
class BaseMergingMethod(ABC):
    """Base class for model merging methods"""

    def __init__(self, **kwargs):
        """
        Initialize merging method

        Args:
            **kwargs: Method-specific parameters
        """
        self.params = kwargs

    @abstractmethod
    def merge(
        self,
        task_vectors: Dict[str, torch.Tensor],
        preference_vector: np.ndarray,
        base_model: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """
        Merge task vectors according to preference vector

        Args:
            task_vectors: Dictionary mapping task names to task vectors
            preference_vector: Preference weights for each task
            base_model: Optional base model for methods that need it

        Returns:
            Merged parameter vector
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"


class BaseTrainingMethod(ABC):
    """
    Base class for multi-task training methods

    Training-based methods fine-tune a model from scratch using multi-task
    training with specialized loss functions (e.g., Chebyshev, EPO, MGDA).

    Unlike parameter merging methods (TIES, Averaging), these methods:
    - Train a new model from the base model
    - Use training data and multi-task optimization
    - Return task vectors (fine-tuned params - base params)

    This class provides common infrastructure for:
    - Loading training data for multiple tasks
    - Initializing models and optimizers
    - Training loops with multi-task batching
    - Computing task vectors from trained models
    """

    def __init__(
        self,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 16,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        normalize_preferences: bool = True,
        use_fp16: bool = False,
        dataloader_num_workers: int = 0,
        max_samples_per_task: Optional[int] = None,
        use_streaming: bool = False,
        **kwargs,
    ):
        """
        Initialize training-based method

        Args:
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            warmup_steps: Number of warmup steps for learning rate scheduler
            weight_decay: Weight decay for regularization
            gradient_accumulation_steps: Number of gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping (0 = no clipping)
            normalize_preferences: Whether to normalize preference vector
            use_fp16: Use mixed precision training (fp16) for memory efficiency
            dataloader_num_workers: Number of workers for DataLoader (0 = main process only)
            max_samples_per_task: Maximum samples per task (None = use all). Reduces RAM usage.
            use_streaming: Use streaming datasets (loads data on-the-fly, minimal RAM)
            **kwargs: Additional method-specific parameters
        """
        self.params = kwargs
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.normalize_preferences = normalize_preferences
        self.use_fp16 = use_fp16
        self.dataloader_num_workers = dataloader_num_workers
        self.max_samples_per_task = max_samples_per_task
        self.use_streaming = use_streaming

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self.learning_rate}, epochs={self.num_epochs})"

    def _load_training_data(
        self,
        dataset_configs: Dict,
        tokenizer: Any,
        task_names: list,
        cache_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load training data for all tasks with memory-efficient strategies

        Supports three modes:
        1. Full loading (default): All data in RAM
        2. Sample limiting: Load subset of data (saves RAM)
        3. Streaming: Load data on-the-fly (minimal RAM, experimental)

        Args:
            dataset_configs: Configuration for each dataset
            tokenizer: Tokenizer for preprocessing
            task_names: List of task names to load
            cache_dir: Optional cache directory for datasets

        Returns:
            Dictionary mapping task names to DataLoaders
        """
        from pathlib import Path

        from torch.utils.data import DataLoader
        from transformers import default_data_collator

        from src.data.loaders import load_hf_dataset, preprocess_dataset

        task_dataloaders = {}

        for task_name in task_names:
            task_cfg = dataset_configs[task_name]

            logger.info(f"\nLoading training data for {task_name}...")

            # Load training dataset with caching
            if self.use_streaming:
                # Streaming mode: minimal RAM but slower (experimental)
                logger.info("  Using streaming mode (minimal RAM)")
                dataset = load_hf_dataset(
                    dataset_path=task_cfg.hf_dataset.path,
                    subset=task_cfg.hf_dataset.get("subset", None),
                    split=task_cfg.hf_dataset.split.train,
                    cache_dir=Path(cache_dir) if cache_dir else None,
                    streaming=True,
                )
            else:
                # Regular loading
                dataset = load_hf_dataset(
                    dataset_path=task_cfg.hf_dataset.path,
                    subset=task_cfg.hf_dataset.get("subset", None),
                    split=task_cfg.hf_dataset.split.train,
                    cache_dir=Path(cache_dir) if cache_dir else None,
                    streaming=False,
                )

                # Limit samples if configured
                if self.max_samples_per_task is not None:
                    original_size = len(dataset)
                    if original_size > self.max_samples_per_task:
                        # Use select for reproducible sampling
                        dataset = dataset.select(range(self.max_samples_per_task))
                        logger.info(
                            f"  Limited dataset: {original_size} -> {self.max_samples_per_task} samples "
                            f"(saves ~{(1 - self.max_samples_per_task/original_size)*100:.0f}% RAM)"
                        )

            # Preprocess with caching enabled
            # Datasets library automatically caches preprocessed data
            processed = preprocess_dataset(
                dataset=dataset,
                tokenizer=tokenizer,
                text_column=task_cfg.preprocessing.text_column,
                text_column_2=task_cfg.preprocessing.get("text_column_2", None),
                label_column=task_cfg.preprocessing.label_column,
                max_length=task_cfg.preprocessing.max_length,
                truncation=task_cfg.preprocessing.truncation,
                padding=task_cfg.preprocessing.padding,
            )

            if not self.use_streaming:
                # Set format for PyTorch (more memory efficient)
                # Only works with non-streaming datasets
                processed.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

            # Create DataLoader with memory-efficient settings
            dataloader = DataLoader(
                processed,
                batch_size=self.batch_size,
                shuffle=True if not self.use_streaming else False,  # Streaming doesn't support shuffle
                collate_fn=default_data_collator,
                num_workers=self.dataloader_num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
                persistent_workers=False,  # Don't keep workers alive between epochs
            )

            task_dataloaders[task_name] = dataloader

            if not self.use_streaming:
                logger.info(f"  Loaded {len(processed)} training samples")
                logger.info(f"  {len(dataloader)} batches per epoch")
            else:
                logger.info("  Streaming dataloader created (size unknown)")

        return task_dataloaders

    def _initialize_model(
        self, base_model_id: str, dataset_configs: Dict, device: torch.device, cache_dir: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Initialize model for multi-task training

        Uses maximum num_labels across all tasks for the classification head.
        For more sophisticated multi-task architectures, subclasses can override.

        Args:
            base_model_id: HuggingFace model ID or path
            dataset_configs: Dataset configurations
            device: Device to place model on
            cache_dir: Optional cache directory for model loading

        Returns:
            Initialized model
        """
        from src.models.loaders import load_model

        # Use max num_labels across all tasks
        max_labels = max(cfg.num_labels for cfg in dataset_configs.values())

        logger.info(f"\nInitializing model from {base_model_id}")
        logger.info(f"  Using {max_labels} output labels (max across tasks)")
        if cache_dir:
            logger.info(f"  Cache directory: {cache_dir}")

        model = load_model(
            model_id=base_model_id,
            num_labels=max_labels,
            device=device,
            cache_dir=cache_dir,
        )

        return model

    def _setup_optimizer(
        self, model: torch.nn.Module, train_dataloaders: Dict
    ) -> Tuple[torch.optim.Optimizer, Any]:
        """
        Setup optimizer and learning rate scheduler

        Args:
            model: Model to optimize
            train_dataloaders: Training dataloaders for calculating total steps

        Returns:
            Tuple of (optimizer, scheduler)
        """
        from transformers import get_linear_schedule_with_warmup

        # Create optimizer (use PyTorch's AdamW)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Calculate total training steps (handle streaming dataloaders)
        try:
            min_steps_per_epoch = min(len(dl) for dl in train_dataloaders.values())
        except TypeError:
            # Streaming dataloaders don't support len()
            if self.max_samples_per_task is not None:
                min_steps_per_epoch = self.max_samples_per_task // self.batch_size
            else:
                # Default to a reasonable number for streaming
                min_steps_per_epoch = 10000 // self.batch_size
            logger.info(f"  Streaming mode detected: estimating {min_steps_per_epoch} steps per epoch")

        total_steps = self.num_epochs * min_steps_per_epoch

        logger.info(f"\nOptimizer setup:")
        logger.info(f"  Steps per epoch: {min_steps_per_epoch}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {self.warmup_steps}")

        # Create scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        return optimizer, scheduler

    @abstractmethod
    def _compute_multi_task_loss(
        self,
        task_losses: torch.Tensor,
        preference_vector: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute multi-task loss from individual task losses

        This is the key method that subclasses must implement to define
        their specific multi-task optimization strategy (e.g., Chebyshev, EPO).

        Args:
            task_losses: Tensor of losses for each task, shape (n_tasks,)
            preference_vector: Preference weights, shape (n_tasks,)
            **kwargs: Additional method-specific parameters

        Returns:
            Scalar loss value to optimize
        """
        pass

    def _train_epoch(
        self,
        model: torch.nn.Module,
        task_dataloaders: Dict,
        task_names: list,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        preference_vector: np.ndarray,
        epoch: int,
        scaler: Optional[Any] = None,
        **kwargs,
    ) -> float:
        """
        Train for one epoch with optional mixed precision

        Args:
            model: Model to train
            task_dataloaders: Dataloaders for each task
            task_names: List of task names (sorted)
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            preference_vector: Preference weights for tasks
            epoch: Current epoch number
            scaler: Optional GradScaler for mixed precision training
            **kwargs: Additional parameters passed to _compute_multi_task_loss

        Returns:
            Average loss for the epoch
        """
        model.train()
        total_loss = 0.0
        num_steps = 0

        # Create iterators for each task
        task_iterators = {
            name: iter(dataloader)
            for name, dataloader in task_dataloaders.items()
        }

        # Determine number of steps (use minimum dataloader length)
        # For streaming datasets, we need to handle them differently
        try:
            min_steps = min(len(dl) for dl in task_dataloaders.values())
        except TypeError:
            # Streaming datasets don't support len()
            # Use a reasonable default or calculate from max_samples_per_task
            if self.max_samples_per_task is not None:
                min_steps = self.max_samples_per_task // self.batch_size
            else:
                # Default to a reasonable number for streaming
                min_steps = 10000 // self.batch_size
            logger.info(f"  Streaming mode: using {min_steps} steps per epoch")

        logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
        logger.info(f"  Training steps: {min_steps}")

        step = 0
        while step < min_steps:
            # Get batch from each task and compute losses
            task_losses = []

            # Use autocast for mixed precision
            if scaler is not None:
                from torch.amp import autocast
                autocast_context = autocast('cuda')
            else:
                from contextlib import nullcontext
                autocast_context = nullcontext()

            with autocast_context:
                # Try to get batches from all tasks
                all_batches_available = True
                temp_task_losses = []

                for task_idx, task_name in enumerate(task_names):
                    try:
                        batch = next(task_iterators[task_name])
                    except StopIteration:
                        if self.use_streaming:
                            # For streaming, we've exhausted the data - stop epoch early
                            all_batches_available = False
                            break
                        else:
                            # For regular datasets, restart iterator
                            task_iterators[task_name] = iter(task_dataloaders[task_name])
                            try:
                                batch = next(task_iterators[task_name])
                            except StopIteration:
                                # Dataset is completely empty
                                all_batches_available = False
                                break

                    # Move batch to device
                    batch = {k: v.to(model.device) for k, v in batch.items()}

                    # Forward pass
                    outputs = model(**batch)
                    temp_task_losses.append(outputs.loss)

                # If we couldn't get batches from all tasks, stop
                if not all_batches_available:
                    logger.info(f"  Early stopping at step {step+1} (data exhausted)")
                    break

                task_losses = temp_task_losses

                # Convert task losses to tensor
                losses_tensor = torch.stack(task_losses)

                # Compute multi-task loss (method-specific)
                preference_tensor = torch.tensor(
                    preference_vector, dtype=losses_tensor.dtype, device=losses_tensor.device
                )

                # Move any tensor kwargs to the correct device
                device_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor):
                        device_kwargs[key] = value.to(losses_tensor.device)
                    else:
                        device_kwargs[key] = value

                multi_task_loss = self._compute_multi_task_loss(
                    task_losses=losses_tensor,
                    preference_vector=preference_tensor,
                    **device_kwargs,
                )

            # Backward pass with gradient scaling
            if scaler is not None:
                scaler.scale(multi_task_loss).backward()
            else:
                multi_task_loss.backward()

            # Optimizer step (with gradient accumulation)
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if scaler is not None:
                    # Gradient clipping with scaler
                    if self.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Gradient clipping without scaler
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            total_loss += multi_task_loss.item()
            num_steps += 1
            step += 1

            # Periodic logging
            if step % 100 == 0 or step == min_steps - 1:
                logger.info(
                    f"  Step {step}/{min_steps} - "
                    f"Loss: {multi_task_loss.item():.4f} - "
                    f"Task Losses: {[f'{l.item():.4f}' for l in losses_tensor]}"
                )

        avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
        return avg_loss

    def train(
        self,
        base_model: str,
        dataset_configs: Dict,
        preference_vector: np.ndarray,
        model_cache_dir: Optional[str] = None,
        dataset_cache_dir: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Train model using multi-task learning and return task vector

        This method trains a new model from the base model using multi-task
        optimization, then returns the task vector (fine-tuned - base).

        The returned task vector represents the parameter changes learned
        during training. It can be:
        - Applied to the base model: merged_model = base + task_vector
        - Combined with other task vectors using merging methods
        - Evaluated directly in the benchmark

        Args:
            base_model: Base model ID or path to fine-tune from (e.g., "FacebookAI/roberta-base")
            dataset_configs: Dataset configurations for loading training data
            preference_vector: Preference weights for each task (will be normalized if normalize_preferences=True)
            model_cache_dir: Optional cache directory for loading base models
            dataset_cache_dir: Optional cache directory for loading datasets
            save_path: Optional path to save the trained model

        Returns:
            Task vector (flattened 1D tensor): fine_tuned_params - base_params
            This represents the parameter delta learned during training.
        """
        import copy

        from src.models.loaders import load_tokenizer
        from src.utils.device import get_device

        # Normalize preferences
        if self.normalize_preferences:
            preference_vector = preference_vector / preference_vector.sum()

        logger.info("=" * 80)
        logger.info(f"Starting {self.__class__.__name__}")
        logger.info("=" * 80)
        logger.info(f"Preference vector: {preference_vector}")
        logger.info(f"Num epochs: {self.num_epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Learning rate: {self.learning_rate}")
        if model_cache_dir:
            logger.info(f"Model cache: {model_cache_dir}")
        if dataset_cache_dir:
            logger.info(f"Dataset cache: {dataset_cache_dir}")

        # Get device
        device = get_device()
        logger.info(f"Using device: {device}")

        # 1. Load tokenizer
        tokenizer = load_tokenizer(base_model, cache_dir=model_cache_dir)

        # 2. Load training data for all tasks
        task_names = sorted(dataset_configs.keys())
        train_dataloaders = self._load_training_data(
            dataset_configs, tokenizer, task_names, cache_dir=dataset_cache_dir
        )

        # 3. Initialize model
        model = self._initialize_model(base_model, dataset_configs, device, cache_dir=model_cache_dir)

        # Enable gradient checkpointing for memory efficiency (if available)
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for memory efficiency")

        base_state_dict = copy.deepcopy(model.state_dict())

        # 4. Setup optimizer and scheduler
        optimizer, scheduler = self._setup_optimizer(model, train_dataloaders)

        # 5. Get method-specific training parameters (pass model cache for utopia point computation)
        training_kwargs = self._get_training_kwargs(
            task_names, preference_vector, dataset_configs, cache_dir=model_cache_dir
        )

        # 6. Setup mixed precision training if enabled
        scaler = None
        if self.use_fp16 and torch.cuda.is_available():
            from torch.amp import GradScaler
            scaler = GradScaler('cuda')
            logger.info("Mixed precision training (FP16) enabled")

        # 7. Training loop
        logger.info("\nStarting training...")
        for epoch in range(self.num_epochs):
            avg_loss = self._train_epoch(
                model=model,
                task_dataloaders=train_dataloaders,
                task_names=task_names,
                optimizer=optimizer,
                scheduler=scheduler,
                preference_vector=preference_vector,
                epoch=epoch,
                scaler=scaler,
                **training_kwargs,
            )

            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Avg Loss: {avg_loss:.4f}")

            # Clear cache after each epoch for memory efficiency
            if device.type == "cuda":
                torch.cuda.empty_cache()
                import gc
                gc.collect()

        # 7. Save trained model if requested
        if save_path:
            self._save_trained_model(model, save_path)

        # 8. Compute task vector from trained model
        logger.info("\nComputing task vector from trained model...")
        trained_state_dict = model.state_dict()
        task_vector_dict = {}

        for name, param in trained_state_dict.items():
            if name in base_state_dict:
                task_vector_dict[name] = param.cpu() - base_state_dict[name].cpu()

        # 9. Flatten and return
        from src.benchmarks.poc.run import flatten_task_vector
        flattened = flatten_task_vector(task_vector_dict)

        logger.info(f"{self.__class__.__name__} completed!")
        logger.info("=" * 80)

        return flattened

    def _save_trained_model(self, model: torch.nn.Module, save_path: str) -> None:
        """
        Save trained model to disk

        Saves only the model state dict in an efficient format (safetensors).

        Args:
            model: Trained model to save
            save_path: Path to save the model (should end with .safetensors)
        """
        from pathlib import Path

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nSaving trained model to {save_path}...")

        # Save using safetensors format (more efficient than torch.save)
        try:
            from safetensors.torch import save_file

            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            save_file(state_dict, str(save_path))
            logger.info(f"  ✓ Model saved successfully")
        except ImportError:
            # Fallback to torch.save if safetensors not available
            logger.warning("  safetensors not available, using torch.save instead")
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ Model saved successfully (torch format)")

    def _load_trained_model(
        self,
        model: torch.nn.Module,
        load_path: str,
        device: Optional[torch.device] = None,
    ) -> torch.nn.Module:
        """
        Load trained model from disk

        Args:
            model: Model instance to load weights into
            load_path: Path to load the model from
            device: Device to load model onto

        Returns:
            Model with loaded weights
        """
        from pathlib import Path

        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        logger.info(f"\nLoading trained model from {load_path}...")

        # Try loading with safetensors first
        try:
            from safetensors.torch import load_file

            state_dict = load_file(str(load_path), device=str(device) if device else "cpu")
            model.load_state_dict(state_dict)
            logger.info(f"  ✓ Model loaded successfully (safetensors format)")
        except (ImportError, Exception):
            # Fallback to torch.load
            state_dict = torch.load(load_path, map_location=device if device else "cpu")
            model.load_state_dict(state_dict)
            logger.info(f"  ✓ Model loaded successfully (torch format)")

        return model

    def _get_training_kwargs(
        self, task_names: list, preference_vector: np.ndarray, dataset_configs: Dict, cache_dir: Optional[str] = None
    ) -> Dict:
        """
        Get method-specific kwargs to pass to training loop

        Subclasses can override this to provide additional parameters
        (e.g., utopia point for Chebyshev, Pareto front for EPO)

        Args:
            task_names: List of task names
            preference_vector: Preference weights
            dataset_configs: Dataset configurations
            cache_dir: Optional cache directory for loading models

        Returns:
            Dictionary of kwargs for _train_epoch
        """
        return {}
