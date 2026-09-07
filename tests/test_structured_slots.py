import json
from dataclasses import replace

import pytest
import torch

from model.structured_slot_head import build_slot_head, slot_groups
from utils.slot_config import SlotConfig
from utils.slot_labels import normalize_slot_labels, empty_slot_labels
from utils.slot_loss import slot_loss


@pytest.mark.parametrize("name", [
    "slot_loss_weight", "slot_smooth_l1_beta", "slot_class_max_weight_ratio",
    "slot_q5_std_floor", "slot_label_smoothing", "slot_progress_monotonic_weight",
    "slot_bbox_giou_weight", "slot_q5_consistency_weight", "slot_presence_weight",
    "slot_obstacle_bbox_weight", "slot_risk_weight",
])
@pytest.mark.parametrize("value", [-1, -1.0])
def test_negative_loss_configuration_rejected(name, value):
    with pytest.raises(ValueError, match="invalid"):
        SlotConfig(**{name: value}).validate()


def config(**kwargs):
    return replace(SlotConfig(slot_aux_type="structured_slots_v1", slot_loss_weight=1), **kwargs)


@pytest.mark.parametrize("count", [7, 16, 20, 24])
def test_groups_and_front_input(count):
    groups = slot_groups(count)
    assert list(groups.values())[0][0] == 0
    assert [i for start, stop in groups.values() for i in range(start, stop)] == list(range(count))
    assert all(stop > start for start, stop in groups.values())
    state = torch.get_rng_state().clone()
    head = build_slot_head(32, count, config())
    assert torch.equal(state, torch.get_rng_state())
    queries = torch.randn(2, count + 8, 32, requires_grad=True)
    outputs = head(queries)
    changed = queries.detach().clone()
    changed[:, count:] += 100
    for key, value in head(changed).items():
        torch.testing.assert_close(value, outputs[key], rtol=0, atol=0)
    sum(v.sum() for v in outputs.values()).backward()
    assert queries.grad[:, count:].count_nonzero() == 0


def test_parameter_count_and_outputs():
    counts = [sum(p.numel() for p in build_slot_head(2048, n, config()).parameters()) for n in (7, 16, 20)]
    assert len(set(counts)) == 1 and counts[0] < 200000
    head = build_slot_head(32, 16, config())
    out = head(torch.randn(3, 24, 32))
    shapes = {"Q1": (2,), "Q2": (2, 7), "Q3": (2, 4), "Q4": (2, 4),
              "Q5": (3, 3), "Q6": (2, 2), "Q7": (4,), "Q8": (4,), "Q9": (3, 6)}
    for q, shape in shapes.items():
        assert out[q].shape == (3, *shape)
    for value in (out["Q1"], out["Q6"], out["Q9"][..., 4]):
        assert ((value >= 0) & (value <= 1)).all()
    for boxes in (out["Q3"], out["Q4"], out["Q9"][..., :4]):
        assert ((boxes >= 0) & (boxes <= 1)).all()
        assert (boxes[..., 2:] >= boxes[..., :2]).all()


def test_anchor_and_independent_obstacle_masks():
    raw = {"query_1": {"progress_t": .2, "progress_tK": .5},
           "query_9": {"valid_mask": [1, 0, 1], "obstacles": [
               {"bbox": [0, 0, 1, 1], "risk_score": None}, {},
               {"bbox": None, "risk_score": .8}]}}
    inactive = normalize_slot_labels(json.dumps(raw), is_anchor=False, identity="test ep0 f1")
    assert not any(v.any() for k, v in inactive.items() if k.endswith("mask"))
    active = normalize_slot_labels(raw, is_anchor=True, identity="test ep0 f0")
    assert active["slot_Q9_presence_mask"].tolist() == [True, True, True]
    assert active["slot_Q9_bbox_mask"].tolist() == [True, False, False]
    assert active["slot_Q9_risk_mask"].tolist() == [False, False, True]
    assert active["slot_Q9_presence"].tolist() == [1, 0, 1]
    del raw["query_9"]["valid_mask"]
    assert not normalize_slot_labels(raw, is_anchor=True)["slot_Q9_presence_mask"].any()


def test_unknown_class_and_invalid_values():
    with pytest.raises(ValueError, match="dataset=x episode=2 frame=3"):
        normalize_slot_labels({"query_7": {"gripper_transition_id": "new", "valid_mask": 0}},
                              is_anchor=True, identity="dataset=x episode=2 frame=3")
    with pytest.raises(ValueError, match="query_1"):
        normalize_slot_labels({"query_1": {"progress_t": float("nan"), "progress_tK": None}}, is_anchor=True)
    labels = normalize_slot_labels({"query_1": {"progress_t": None},
                                   "query_7": {"gripper_transition_id": "maintain_open", "valid_mask": 0}}, is_anchor=True)
    assert not labels["slot_Q1_mask"].any() and not labels["slot_Q7_mask"].any()


def stats():
    from utils.slot_labels import VOCABULARIES
    return {"classes": {q: {"vocabulary": vocab, "weights": [1.] * len(vocab)} for q, vocab in VOCABULARIES.items()},
            "q5_mean": [[0.] * 3] * 3, "q5_std": [[1.] * 3] * 3}


def test_empty_loss_connects_every_parameter():
    head = build_slot_head(16, 7, config())
    x = torch.randn(2, 15, 16, requires_grad=True)
    labels = {k: torch.stack([v, v]) for k, v in empty_slot_labels().items()}
    out = slot_loss(head(x), labels, config(), stats())
    assert out["slot_loss_raw"].item() == 0
    out["slot_loss_raw"].backward()
    assert x.grad is not None
    assert all(p.grad is not None and p.grad.count_nonzero() == 0 for p in head.parameters())


def test_progress_hand_calculation():
    head = build_slot_head(16, 7, config())
    predictions = head(torch.randn(1, 7, 16))
    predictions["Q1"] = torch.tensor([[.8, .2]], requires_grad=True)
    labels = normalize_slot_labels({"query_1": {"progress_t": .2, "progress_tK": .6}}, is_anchor=True)
    labels = {k: v.unsqueeze(0) for k, v in labels.items()}
    out = slot_loss(predictions, labels, config(), stats())
    assert out["slot_Q1_loss_sum"].item() == pytest.approx((.6**2 / 2 + .4**2 / 2) / 2 + .1 * .6)
    assert out["slot_Q1_count"].item() == 1


@pytest.mark.parametrize("q,width", [("Q2", 7), ("Q7", 4), ("Q8", 4)])
def test_uniform_classification_hand_calculation(q, width):
    import math
    predictions = build_slot_head(16, 7, config())(torch.randn(1, 7, 16))
    predictions[q] = torch.zeros_like(predictions[q], requires_grad=True)
    labels = {k: v.unsqueeze(0) for k, v in empty_slot_labels().items()}
    labels[f"slot_{q}_mask"].fill_(True)
    statistics = stats()
    statistics["classes"][q]["weights"] = list(range(1, width + 1))
    result = slot_loss(predictions, labels, config(), statistics)
    expected = math.log(width) * (.98 + .02 * (width + 1) / 2)
    assert result[f"slot_{q}_loss_sum"].item() == pytest.approx(expected)


@pytest.mark.parametrize("q", ["Q3", "Q4"])
def test_bbox_hand_calculation(q):
    predictions = build_slot_head(16, 7, config())(torch.randn(1, 7, 16))
    predictions[q] = torch.tensor([[[0., 0., 1., 1.], [0., 0., 1., 1.]]], requires_grad=True)
    labels = {k: v.unsqueeze(0) for k, v in empty_slot_labels().items()}
    labels[f"slot_{q}"][:, 0] = torch.tensor([.25, .25, .75, .75])
    labels[f"slot_{q}_mask"][:, 0] = True
    result = slot_loss(predictions, labels, config(), stats())
    assert result[f"slot_{q}_loss_sum"].item() == pytest.approx(.25**2 / 2 + .1 * .75)


def test_q5_normalization_and_metric_consistency_hand_calculation():
    predictions = build_slot_head(16, 7, config())(torch.randn(1, 7, 16))
    predictions["Q5"] = torch.zeros(1, 3, 3, requires_grad=True)
    labels = {k: v.unsqueeze(0) for k, v in empty_slot_labels().items()}
    labels["slot_Q5_mask"].fill_(True)
    statistics = stats()
    statistics["q5_mean"] = [[0., 0., 0.], [0., 0., 0.], [1., 1., 1.]]
    statistics["q5_std"] = [[2.] * 3] * 3
    result = slot_loss(predictions, labels, config(), statistics)
    assert result["slot_Q5_loss_sum"].item() == pytest.approx(.5**2 / 2 / 3 + .05 * .5)
    labels["slot_Q5_mask"][:, 0] = False
    result = slot_loss(predictions, labels, config(), statistics)
    assert result["slot_Q5_loss_sum"].item() == pytest.approx(.5**2 / 2 / 2)


def test_q6_and_q9_hand_calculation_with_disjoint_masks():
    import math
    predictions = build_slot_head(16, 7, config())(torch.randn(1, 7, 16))
    predictions["Q6"] = torch.full((1, 2, 2), .5, requires_grad=True)
    predictions["Q9"] = torch.tensor([[[0., 0., 1., 1., .5, 0.]] * 3], requires_grad=True)
    labels = {k: v.unsqueeze(0) for k, v in empty_slot_labels().items()}
    labels["slot_Q6_mask"][:, 1] = True
    labels["slot_Q9_presence_mask"][:, 0] = True
    labels["slot_Q9_bbox_mask"][:, 1] = True
    labels["slot_Q9_bbox"][:, 1] = torch.tensor([.25, .25, .75, .75])
    labels["slot_Q9_risk_mask"][:, 2] = True
    result = slot_loss(predictions, labels, config(), stats())
    assert result["slot_Q6_loss_sum"].item() == pytest.approx(.125)
    expected = .5 * math.log(2) + .25 * (.25**2 / 2 + .075) + .25 * .125
    assert result["slot_Q9_loss_sum"].item() == pytest.approx(expected)
    result["slot_loss_raw"].backward()
    assert predictions["Q9"].grad[0, 0, 4] == 0
    assert predictions["Q9"].grad[0, 2, 4] != 0


def test_collator_fills_only_missing_masks():
    from utils.load_training_dataset import custom_collate_fn
    first = {"input_ids": torch.tensor([1]), **normalize_slot_labels({"query_1": {"progress_t": .2}}, is_anchor=True)}
    second = {"input_ids": torch.tensor([2])}
    batch = custom_collate_fn([first, second])
    assert batch["slot_Q1_mask"].tolist() == [[True, False], [False, False]]


def test_cli_slot_layout_and_checkpointless_defaults():
    from utils.cli_options import parse_train_options
    from utils.slot_config import resolve_query_layout
    assert parse_train_options([]).slot_aux_type == "none"
    base = ["--training_stage", "stage2_aux", "--slot_aux_type", "structured_slots_v1", "--slot_loss_weight", "1",
            "--slot_supervision_dir", "/fixture", "--init_from_checkpoint", "/fixture",
            "--use_difference_query", "--num_difference_queries", "32"]
    with pytest.raises(SystemExit):
        parse_train_options(base)
    options = parse_train_options(base + ["--num_flow_queries", "8"])
    assert not options.tune_vlm and not options.tune_action_expert
    assert options.loss_type == "aux"
    joint = list(base)
    joint[joint.index("stage2_aux")] = "stage3_joint"
    assert parse_train_options(joint + ["--num_flow_queries", "8", "--tune_action_expert"]).tune_vlm
    assert resolve_query_layout(7, 0, slot_enabled=True, query_enabled=True)["num_slot_queries"] == 7
