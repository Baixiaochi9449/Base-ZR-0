# LIBERO Evaluation Runbook Design

## Goal

Turn `simple_scripts/eval_libero.md` into a copy-paste runbook for evaluating the
official ZR-0-LIBERO checkpoint on the current server. The runbook assumes the
model, repository, LIBERO checkout, and both Conda environments already exist.

## Scope

The document covers only direct evaluation with the existing installation. It
does not cover downloading the model, cloning LIBERO, creating environments, or
installing dependencies.

## Structure

1. List the fixed repository, model, environment, and result paths.
2. Check required files, GPU availability, and the four evaluation ports.
3. Create a timestamped result directory so an earlier run is not overwritten.
4. Start one ZR-0 policy server per LIBERO suite on GPUs 0-3.
5. Check that each server loaded its checkpoint and is listening.
6. Start the four headless LIBERO clients with the repository defaults.
7. Monitor logs, wait for completion, and extract final success rates.
8. Stop policy servers and point to logs, videos, and the previous verified run.
9. Record the default evaluation parameters without presenting them as command
   overrides.

## Runtime Mapping

| Suite | GPU | Port |
| --- | ---: | ---: |
| `libero_spatial` | 0 | 8100 |
| `libero_object` | 1 | 8001 |
| `libero_goal` | 2 | 8002 |
| `libero_10` | 3 | 8103 |

The ports match the completed verified run. The runbook checks that they are
free before launching and requires the server and client mappings to stay in
sync if a port is changed.

## Evaluation Defaults

The commands select only the suite, matching server port, model checkpoint, and
output directory. Repository defaults remain unchanged: direct action mode,
server seed 42, client seed 7, window size 1, action chunk/replan horizon 10,
five denoising steps, 448-pixel model inputs, 256-pixel simulator rendering, ten
stabilization steps, and 50 trials per task. Suite horizons remain defined in
`evaluation/libero_eval/run_libero_eval.py`.

## Validation

- Every shell block is syntactically valid Bash and uses existing absolute
  paths on the current server.
- The documented Tyro option names match `run_libero_eval.py --help`.
- GPU, port, suite, log, and video mappings are consistent throughout.
- The runbook never writes into the previous verified result directory.
- Environment setup and dependency installation remain out of scope.
