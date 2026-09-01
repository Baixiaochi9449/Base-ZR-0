# ZR-0 Gradient Explosion Fix Design

## Objective

Make the LIBERO `w/o ECoT PT` ablation numerically stable without changing its
experimental meaning: initialize the VLM from Qwen3-VL-2B-Instruct, randomly
initialize the action expert, train both modules directly on public LIBERO v2.1
data, use global batch size 64, action horizon 10, action loss weight 1, and the
existing post-training optimizer and scheduler settings.

## Confirmed Root Cause

`QwenVLBackbone.forward()` currently sends
`model_outputs["hidden_states"][-1]` directly to the action expert. With the
pinned Transformers 4.57.1 implementation, the outer
`Qwen3VLForConditionalGeneration` output recorder captures the output of the
last `Qwen3VLTextDecoderLayer` before Qwen's final RMSNorm. The language-model
head itself consumes the final-RMSNorm output.

The two representations were compared in the same Qwen forward pass and were
proven to satisfy:

```text
language_model.norm(model_outputs.hidden_states[-1])
    == captured final RMSNorm output
```

The equality was exact in BF16. On a real LIBERO sample, the recorded feature
RMS grew from 54.55 in the base model to 4328.90 at step 18000 and 5187.33 at
step 20000, while the final-RMSNorm feature stayed near 2-3. In a controlled
step-20000 action-expert backward pass, changing only this representation
reduced the action-expert gradient norm from 639.62 to 29.91.

The action and state dataset scan found no NaN or Inf values and no values that
reached the normalization clamp at +/-15. The data normalization path is not
the trigger.

## Selected Approach

### Correct the VLM-to-action-expert interface

Keep the existing Qwen conditional-generation forward pass so VLM loss and
generation behavior remain unchanged. Before returning `backbone_embeddings`,
apply Qwen's own final text RMSNorm to the captured last-decoder output. This is
the smallest change that produces the same normalized representation consumed
by Qwen's language-model head, and it preserves gradient flow from the action
loss into the VLM.

The implementation must work for both a plain Qwen model and the repository's
optional PEFT wrapper. It must not detach or freeze the VLM.

### Stop immediately on non-finite training state

Add a lightweight validation helper around each completed optimizer step:

- Reject a non-finite loss before backward.
- Read DeepSpeed's already-computed global gradient norm after backward/step.
- Reject a non-finite global gradient norm before updating counters or saving a
  checkpoint.

The BF16 ZeRO-2 path in DeepSpeed 0.15.4 does not run the FP16 overflow check.
The post-step gradient check cannot undo the in-memory update that produced the
non-finite value, but immediate termination prevents later checkpoints from
being overwritten with corrupted weights. Periodic finite checkpoints remain
the recovery boundary. The final RMSNorm correction addresses the actual
cause; this check is defense in depth.

Do not introduce an arbitrary finite-gradient threshold. Large but finite
gradients remain governed by the existing global gradient clipping value of 1.

## Rejected Alternatives

- Lowering the learning rate, freezing the VLM, or detaching VLM features would
  change the ablation and only hide the interface error.
- Reimplementing the complete Qwen conditional-generation forward to expose
  `last_hidden_state` would duplicate label-loss and multimodal forwarding logic
  and create a larger compatibility surface.
- Resuming from step 18000 would combine a corrected feature interface with an
  action expert trained for the old unnormalized distribution.

## Testing

Follow test-driven development:

1. Add a unit regression test showing that the backbone returns the final
   RMSNorm representation rather than the raw last-decoder output. Cover both
   plain and PEFT-style model access without loading multi-billion-parameter
   weights.
2. Add unit tests showing that finite loss/gradient values pass and NaN or Inf
   values raise before checkpoint logic can run.
3. Run the existing focused launcher and W&B logger tests.
4. Run the complete lightweight test suite available in the repository.
5. Repeat the real-sample feature equality and gradient diagnostic on GPU.

## Training Validation And Rollout

All training validation starts from Qwen3-VL-2B-Instruct and a newly randomized
action expert with seed 42. No existing action-expert checkpoint is loaded.

1. Run a short four-GPU smoke training with the same global batch size 64,
   optimizer, scheduler, BF16, ZeRO-2, action horizon 10, and loss configuration
   as the full experiment. Only the maximum step count/output location may be
   shortened or isolated for the smoke run.
2. Verify throughout the smoke run that loss and global gradient norm are
   finite, checkpoint tensors are finite, and the Qwen features passed to the
   action expert stay normalized.
3. Only after the smoke gate passes, start the complete eight-epoch experiment
   from scratch in a fresh output directory and a fresh W&B run.
4. Preserve the failed run and its checkpoints as diagnostic evidence; never
   resume from or overwrite them.

## Repository And Git Constraints

All changes for this fix are staged but not committed. Existing user changes
and the failed training artifacts are preserved.
