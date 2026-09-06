import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.audit_stage05_token_lengths import (
    DATASET_KEYS_TO_ENTRIES,
    TOKENIZATION_IMPLEMENTATION_PATHS,
    TOKEN_AUDIT_FORMAT_VERSION,
    _content_hash,
    _data_identities,
    _package_versions,
    _processor_files,
    _repository_implementation_identity,
    validate_token_audit,
)
from utils.stage05_sidecar import (
    GENERATOR_COMMON_DEPENDENCY_PATHS,
    canonical_json_hash,
    generator_identity,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_fixture(tmp_path: Path):
    repository = tmp_path / "repository"
    for relative in dict.fromkeys(
        (*TOKENIZATION_IMPLEMENTATION_PATHS, *GENERATOR_COMMON_DEPENDENCY_PATHS)
    ):
        source = ROOT / relative
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    processor = tmp_path / "processor"
    processor.mkdir()
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "config.json",
        "chat_template.json",
    ):
        (processor / name).write_text(f"{name} fixture\n", encoding="utf-8")

    registry = {}
    for index, (key, entry_name) in enumerate(DATASET_KEYS_TO_ENTRIES.items()):
        sidecar = tmp_path / "sidecars" / "ar" / key
        sidecar.mkdir(parents=True)
        eligible = np.asarray([index, index + 4], dtype=np.uint32)
        np.save(sidecar / "ar_indices.npy", eligible, allow_pickle=False)
        entry = {
            "dataset_path": str(tmp_path / "datasets" / key),
            "ar_sidecar_path": str(sidecar),
            "stats_key": f"stage05_{key}",
        }
        registry[entry_name] = entry
        manifest = {
            "sidecar_format_version": 2,
            "dataset_id": entry["stats_key"],
            "generator_identity": generator_identity(
                repository_root=repository, build_joint=False
            ),
            "generation": {"build_joint": False},
            "counts": {"ar_eligible_frames": len(eligible)},
            "files": {"ar_indices.npy": sha256_file(sidecar / "ar_indices.npy")},
        }
        manifest["content_hash"] = canonical_json_hash(manifest)
        _write_json(sidecar / "manifest.json", manifest)
    (repository / "dataset2feature.yaml").write_text(
        yaml.safe_dump(registry, sort_keys=True), encoding="utf-8"
    )

    source_path = ROOT / "utils/training_tokenization.py"
    source_record = {
        "path": str(source_path.resolve()),
        "sha256": sha256_file(source_path),
        "size": source_path.stat().st_size,
    }
    report = {
        "token_audit_format_version": TOKEN_AUDIT_FORMAT_VERSION,
        "data_identity": _data_identities(repository_root=repository),
        "processor_identity": {
            "audited_processor_path": str(processor.resolve()),
            "processor_class": "fixture.Processor",
            "tokenizer_class": "fixture.Tokenizer",
            "image_processor_class": "fixture.ImageProcessor",
            "package_versions": _package_versions(),
            "model_files": _processor_files(processor),
            "runtime_source_files": [source_record],
            "tokenizer_parameters": {"model_max_length": 4096},
            "visual_token_parameters": {"patch_size": 16, "merge_size": 2},
        },
        "implementation_identity": _repository_implementation_identity(repository),
        "vision_contract": {
            "camera_order": ["main camera", "wrist camera"],
            "valid_image_counts": [1, 2],
            "resize": {"height": 224, "width": 224},
            "processor_measured_visual_tokens": {"1": {}, "2": {}},
        },
        "required_max_length": 941,
    }
    report["content_hash"] = _content_hash(report)
    report_path = tmp_path / "token-audit.json"
    _write_json(report_path, report)
    return repository, processor, report_path


def _rewrite_report(path: Path, mutate, *, refresh_hash=True):
    report = json.loads(path.read_text(encoding="utf-8"))
    mutate(report)
    if refresh_hash:
        report["content_hash"] = _content_hash(report)
    _write_json(path, report)


def test_current_v2_identity_and_length_boundaries_validate(tmp_path):
    repository, processor, report = _make_fixture(tmp_path)
    assert validate_token_audit(
        report, processor_path=processor, max_length=941, repository_root=repository
    ) == 941
    assert validate_token_audit(
        report, processor_path=processor, max_length=1024, repository_root=repository
    ) == 941
    with pytest.raises(ValueError, match="below the audited Stage05 minimum 941"):
        validate_token_audit(
            report, processor_path=processor, max_length=940, repository_root=repository
        )


def test_external_trusted_spec_binds_report_and_implementation(tmp_path):
    repository, processor, report = _make_fixture(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    spec = tmp_path / "stage05-spec.json"
    _write_json(
        spec,
        {
            "schema_version": 1,
            "experiment": "fixture",
            "token_audit": {
                "format_version": TOKEN_AUDIT_FORMAT_VERSION,
                "required_max_length": 941,
                "report_content_hash": payload["content_hash"],
                "report_file_sha256": sha256_file(report),
                "implementation_identity_sha256": payload["implementation_identity"]["sha256"],
            },
        },
    )
    assert validate_token_audit(
        report,
        processor_path=processor,
        max_length=941,
        repository_root=repository,
        trusted_spec_path=spec,
    ) == 941

    _rewrite_report(report, lambda value: value.__setitem__("required_max_length", 940))
    with pytest.raises(ValueError, match="report file hash differs|content hash differs"):
        validate_token_audit(
            report,
            processor_path=processor,
            max_length=941,
            repository_root=repository,
            trusted_spec_path=spec,
        )


def test_sidecar_manifest_identity_change_fails(tmp_path):
    repository, processor, report = _make_fixture(tmp_path)
    sidecar_manifest = (
        tmp_path / "sidecars/ar/droid/manifest.json"
    )
    value = json.loads(sidecar_manifest.read_text(encoding="utf-8"))
    value["review_mutation"] = True
    value["content_hash"] = canonical_json_hash(
        {key: item for key, item in value.items() if key != "content_hash"}
    )
    _write_json(sidecar_manifest, value)
    with pytest.raises(ValueError, match="sidecar.*identity changed"):
        validate_token_audit(
            report, processor_path=processor, max_length=941, repository_root=repository
        )


def test_eligible_index_identity_change_fails(tmp_path):
    repository, processor, report = _make_fixture(tmp_path)
    index_path = tmp_path / "sidecars/ar/droid/ar_indices.npy"
    np.save(index_path, np.asarray([0, 5], dtype=np.uint32), allow_pickle=False)
    manifest_path = index_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["ar_indices.npy"] = sha256_file(index_path)
    manifest["content_hash"] = canonical_json_hash(
        {key: item for key, item in manifest.items() if key != "content_hash"}
    )
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="sidecar.*identity changed"):
        validate_token_audit(
            report, processor_path=processor, max_length=941, repository_root=repository
        )


@pytest.mark.parametrize(
    "filename", ["tokenizer.json", "chat_template.json", "preprocessor_config.json"]
)
def test_processor_tokenization_file_change_fails(tmp_path, filename):
    repository, processor, report = _make_fixture(tmp_path)
    with (processor / filename).open("a", encoding="utf-8") as output:
        output.write("changed\n")
    with pytest.raises(ValueError, match="processor/tokenizer files identity changed"):
        validate_token_audit(
            report, processor_path=processor, max_length=941, repository_root=repository
        )


@pytest.mark.parametrize(
    "relative",
    [
        "utils/dataset_adapters.py",
        "utils/stage05_dataset.py",
        "utils/training_tokenization.py",
    ],
)
def test_message_and_tokenization_source_change_fails(tmp_path, relative):
    repository, processor, report = _make_fixture(tmp_path)
    with (repository / relative).open("a", encoding="utf-8") as output:
        output.write("# token audit identity mutation\n")
    with pytest.raises(ValueError, match="stale"):
        validate_token_audit(
            report, processor_path=processor, max_length=941, repository_root=repository
        )


def test_old_format_and_missing_identity_fields_fail(tmp_path):
    repository, processor, report = _make_fixture(tmp_path)
    _rewrite_report(
        report, lambda value: value.__setitem__("token_audit_format_version", 1)
    )
    with pytest.raises(ValueError, match="format version"):
        validate_token_audit(
            report, processor_path=processor, max_length=941, repository_root=repository
        )

    repository, processor, report = _make_fixture(tmp_path / "missing")
    _rewrite_report(report, lambda value: value.pop("processor_identity"))
    with pytest.raises(ValueError, match="identity fields are missing"):
        validate_token_audit(
            report, processor_path=processor, max_length=941, repository_root=repository
        )


def test_report_tampering_with_or_without_updated_self_hash_still_fails(tmp_path):
    repository, processor, report = _make_fixture(tmp_path)
    _rewrite_report(
        report,
        lambda value: value["data_identity"][0].__setitem__(
            "manifest_content_hash", "0" * 64
        ),
        refresh_hash=False,
    )
    with pytest.raises(ValueError, match="report content hash mismatch"):
        validate_token_audit(
            report, processor_path=processor, max_length=941, repository_root=repository
        )

    repository, processor, report = _make_fixture(tmp_path / "rehash")
    _rewrite_report(
        report,
        lambda value: value["data_identity"][0].__setitem__(
            "manifest_content_hash", "0" * 64
        ),
        refresh_hash=True,
    )
    with pytest.raises(ValueError, match="sidecar.*identity changed"):
        validate_token_audit(
            report, processor_path=processor, max_length=941, repository_root=repository
        )
