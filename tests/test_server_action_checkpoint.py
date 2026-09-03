from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("checkpoint_kind", ["action_only", "joint"])
def test_server_delegates_action_capable_checkpoint_to_policy(monkeypatch, checkpoint_kind):
    import server as server_module

    checkpoint = f"/tmp/{checkpoint_kind}"
    options = SimpleNamespace(
        dataset_entry="fixture",
        ckpt_dir=checkpoint,
        inference_mode="direct_action",
        window_size=1,
        num_denoised_steps=5,
        max_pad_state_and_action_length=64,
        use_difference_query=True,
        num_difference_queries=32,
        vlm_attention_backend="sdpa",
        allow_legacy_checkpoint_without_manifest=False,
        port=8000,
    )
    policy_factory = Mock(return_value=object())
    serving = Mock()
    server_factory = Mock(return_value=serving)
    monkeypatch.setattr(server_module, "parse_option", lambda: options)
    monkeypatch.setattr(server_module, "set_all_seeds", lambda *_: None)
    monkeypatch.setattr(server_module, "ZR0Policy", policy_factory)
    monkeypatch.setattr(server_module, "WebsocketPolicyServer", server_factory)

    server_module.deploy()

    assert policy_factory.call_args.kwargs["ckpt_dir"] == checkpoint
    serving.serve_forever.assert_called_once_with()


def test_server_surfaces_ar_only_action_inference_error(monkeypatch):
    import server as server_module

    options = SimpleNamespace(
        dataset_entry="fixture",
        ckpt_dir="/tmp/ar-only",
        inference_mode="direct_action",
        window_size=1,
        num_denoised_steps=5,
        max_pad_state_and_action_length=64,
        use_difference_query=True,
        num_difference_queries=32,
        vlm_attention_backend="sdpa",
        allow_legacy_checkpoint_without_manifest=False,
        port=8000,
    )
    monkeypatch.setattr(server_module, "parse_option", lambda: options)
    monkeypatch.setattr(server_module, "set_all_seeds", lambda *_: None)
    monkeypatch.setattr(
        server_module,
        "ZR0Policy",
        lambda **_: (_ for _ in ()).throw(
            ValueError("ar_only cannot be used for action inference")
        ),
    )
    with pytest.raises(ValueError, match="ar_only.*action inference"):
        server_module.deploy()
