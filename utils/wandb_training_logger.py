import logging


class WandbTrainingLogger:
    def __init__(
        self,
        accelerator,
        *,
        project,
        run_name,
        run_id,
        resume,
        log_dir,
        group=None,
        tags=None,
        config=None,
    ):
        self.accelerator = accelerator
        self.enabled = bool(project)
        self.run = None
        self._pending = []

        if self.enabled and accelerator.is_main_process:
            import wandb

            self.run = wandb.init(
                project=project,
                name=run_name,
                id=run_id,
                resume=resume,
                dir=log_dir,
                group=group,
                tags=tags,
                config=config,
                mode="online",
                save_code=True,
            )

    def log(self, *, step, mean_metrics, scalar_metrics):
        if not self.enabled:
            return

        reduced_metrics = {}
        for name, value in mean_metrics.items():
            reduced_value = self.accelerator.reduce(
                value.detach().float(), reduction="mean"
            )
            if self.accelerator.is_main_process:
                reduced_metrics[name] = reduced_value.item()

        if self.run is not None:
            payload = ({**reduced_metrics, **scalar_metrics}, step)
            pending = [*self._pending, payload]
            self._pending = []
            for index, (metrics, pending_step) in enumerate(pending):
                try:
                    self.run.log(metrics, step=pending_step)
                except Exception as error:
                    self._pending.extend(pending[index:])
                    logging.getLogger(__name__).warning(
                        "W&B logging failed after training started; retained %d "
                        "payload(s) for retry while local logs remain authoritative: %s",
                        len(self._pending),
                        error,
                    )
                    break

    def finish(self, exit_code=0):
        if self.run is not None:
            if self._pending:
                pending = self._pending
                self._pending = []
                for index, (metrics, pending_step) in enumerate(pending):
                    try:
                        self.run.log(metrics, step=pending_step)
                    except Exception as error:
                        self._pending.extend(pending[index:])
                        logging.getLogger(__name__).warning(
                            "W&B final retry failed; %d payload(s) remain only in "
                            "authoritative local logs: %s",
                            len(self._pending),
                            error,
                        )
                        break
            try:
                self.run.finish(exit_code=exit_code)
            except Exception as error:
                logging.getLogger(__name__).warning(
                    "W&B finish failed after local training state was saved: %s",
                    error,
                )
