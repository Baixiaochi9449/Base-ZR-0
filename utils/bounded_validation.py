"""Explicit successful-update limits, independent of scheduler length."""

VALIDATION_UPDATES = 100
VALIDATION_MILESTONES = (50, 100)


def validate_update_limits(options):
    stop = getattr(options, "save_and_exit_after_updates", None)
    skips = getattr(options, "max_consecutive_skipped_windows", None)
    bounded = getattr(options, "bounded_three_stage_validation", False)
    component_diagnostics = getattr(options, "component_update_diagnostics", None)
    if component_diagnostics is not None and not getattr(options, "component_optimizer_groups", False):
        raise ValueError("component diagnostics require component optimizer groups")
    if getattr(options, "verify_resume_state", False) and (
            not getattr(options, "resume_training", False) or not getattr(options, "training_stage", None)):
        raise ValueError("resume-state verification requires a same-stage resume")
    if getattr(options, "fast_resume_data_skip", False) and not getattr(options, "training_stage", None):
        raise ValueError("fast data resume requires an explicit three-stage dataset contract")
    if getattr(options, "verify_three_stage_initialization", False) and options.training_stage not in {
            "stage1_ar", "stage2_aux", "stage3_joint"}:
        raise ValueError("three-stage initialization verification requires an explicit stage")
    if stop is not None and (type(stop) is not int or stop < 1):
        raise ValueError("save_and_exit_after_updates must be positive")
    if skips is not None and (type(skips) is not int or skips < 1):
        raise ValueError("max_consecutive_skipped_windows must be positive")
    if bounded:
        if options.training_stage not in {"stage1_ar", "stage2_aux", "stage3_joint"}:
            raise ValueError("bounded validation requires an explicit stage")
        if options.max_train_steps != VALIDATION_UPDATES or stop not in VALIDATION_MILESTONES:
            raise ValueError("bounded validation requires scheduler length 100 and save/exit at 50 or 100")
        if skips != 20 or not options.save_optimizer_and_lr_states:
            raise ValueError("bounded validation requires full checkpoints and skip limit 20")
        if not options.component_optimizer_groups:
            raise ValueError("bounded validation requires component optimizer groups")
        if component_diagnostics == "inactive":
            raise ValueError("bounded validation requires full component diagnostics")
        if not options.wandb_project or options.wandb_failure_policy != "required":
            raise ValueError("bounded validation requires connected W&B")
    if getattr(options, "component_optimizer_groups", False) and not options.training_stage:
        raise ValueError("component optimizer groups require an explicit stage")


def execution_limit(options, scheduler_steps):
    validate_update_limits(options)
    limits = [scheduler_steps]
    if getattr(options, "save_and_exit_after_updates", None) is not None:
        limits.append(options.save_and_exit_after_updates)
    if getattr(options, "bounded_three_stage_validation", False):
        limits.append(VALIDATION_UPDATES)
    return min(limits)


def check_next_update(completed, limit):
    if completed >= limit:
        raise RuntimeError(f"optimizer update budget exhausted: {completed}/{limit}")


def consecutive_skips(previous, applied, limit):
    count = 0 if applied else previous + 1
    if limit is not None and count >= limit:
        raise RuntimeError(f"stopped after {count} consecutive windows without a successful optimizer update")
    return count
