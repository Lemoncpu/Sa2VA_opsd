from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper


class OldPolicySyncHook(Hook):
    """Marks the iter boundary for GRPO old-policy semantics without model copying."""

    priority = "VERY_HIGH"

    @staticmethod
    def _unwrap_model(runner):
        model = runner.model
        return model.module if is_model_wrapper(model) else model

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        del batch_idx, data_batch
        model = self._unwrap_model(runner)
        sync_fn = getattr(model, "_sync_old_policy", None)
        if callable(sync_fn):
            sync_fn()
