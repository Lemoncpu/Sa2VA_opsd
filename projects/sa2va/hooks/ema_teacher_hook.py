from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper


class EMATeacherHook(Hook):
    """Updates the teacher after the student optimizer step."""

    priority = "NORMAL"

    @staticmethod
    def _unwrap_model(runner):
        model = runner.model
        return model.module if is_model_wrapper(model) else model

    @staticmethod
    def _did_optimizer_step(runner, batch_idx):
        optim_wrapper = getattr(runner, "optim_wrapper", None)
        if optim_wrapper is None:
            return True

        accumulative_counts = int(
            getattr(
                optim_wrapper,
                "_accumulative_counts",
                getattr(optim_wrapper, "accumulative_counts", 1),
            )
            or 1
        )
        if accumulative_counts <= 1:
            return True

        inner_count = getattr(optim_wrapper, "_inner_count", None)
        if inner_count is None:
            return True
        if inner_count % accumulative_counts == 0:
            return True

        dataloader = getattr(getattr(runner, "train_loop", None), "dataloader", None)
        try:
            num_batches = len(dataloader)
        except Exception:
            num_batches = None
        return num_batches is not None and (batch_idx + 1) >= num_batches

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        del data_batch, outputs
        if not self._did_optimizer_step(runner, batch_idx):
            return

        model = self._unwrap_model(runner)
        update_fn = getattr(model, "update_teacher_ema", None)
        if callable(update_fn):
            update_fn()
