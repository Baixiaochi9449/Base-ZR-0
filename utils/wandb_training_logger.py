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
            self.run.log({**reduced_metrics, **scalar_metrics}, step=step)

    def finish(self, exit_code=0):
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
