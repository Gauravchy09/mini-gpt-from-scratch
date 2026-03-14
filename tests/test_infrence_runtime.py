from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from infrence.runtime import find_export_dirs, pick_default_export, remap_state_dict_keys


def test_remap_state_dict_keys() -> None:
    state_dict = {
        "embedding.token_emb.weight": torch.zeros((2, 2)),
        "embedding.pos_emb.weight": torch.ones((2, 2)),
        "lm_head.weight": torch.randn((2, 2)),
    }

    remapped = remap_state_dict_keys(state_dict)

    assert "embedding.token_embedding.weight" in remapped
    assert "embedding.position_embedding.weight" in remapped
    assert "embedding.token_emb.weight" not in remapped
    assert "embedding.pos_emb.weight" not in remapped
    assert "lm_head.weight" in remapped


def test_find_export_dirs_prefers_run02(tmp_path: Path) -> None:
    run01 = tmp_path / "run_01-abc" / "run_01"
    run02 = tmp_path / "run_02-xyz" / "run_02"

    for target in [run01, run02]:
        target.mkdir(parents=True, exist_ok=True)
        (target / "mini_gpt_state.pt").write_text("x", encoding="utf-8")

    export_dirs = find_export_dirs(tmp_path)

    assert export_dirs
    assert export_dirs[0].name == "run_02"
    assert pick_default_export(export_dirs) == export_dirs[0]
