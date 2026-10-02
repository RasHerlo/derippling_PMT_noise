"""Destination mapping, write-guard, sparse mask roundtrip, dest-only v4 write."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "reference" / "gpt"))

from batch_defringe.discover import job_for_stack  # noqa: E402
from batch_defringe.process_v4 import grow_frame, process_stack_v4  # noqa: E402
from batch_defringe.v4_export import (  # noqa: E402
    assert_writable,
    copy_experiment_xml,
    densify,
    forbidden_root_for,
    is_drive_io_error,
    load_mask_patterns,
    save_mask_patterns,
    slice_sparse,
    sparsify,
    v4_dest_dir,
)

H, W = 128, 128
XML = """<?xml version="1.0"?>
<ThorImageExperiment>
  <Computer name="TEST_PC"/>
  <LSM frameRate="15.136" pixelX="128" pixelY="128" fieldSize="178"
       pixelSizeUM="1.6" flybackCycles="12" scanMode="1" twoWayAlignment="0"
       averageMode="0" averageNum="1"/>
  <PMT gainA="1" gainB="1"/>
  <Magnification mag="16"/>
  <Date date="2024-09-17"/>
</ThorImageExperiment>
"""


def _vstripes(q: float = 8.0) -> np.ndarray:
    x = np.arange(W)[None, :]
    return np.broadcast_to(80.0 + 40.0 * np.sin(2.0 * np.pi * q * x / W), (H, W)).copy()


def test_is_drive_io_error():
    assert is_drive_io_error(OSError("262144 requested and 0 written"))
    assert is_drive_io_error(PermissionError("denied"))
    src = Path("C:/Windows")
    assert not is_drive_io_error(
        FileNotFoundError("missing.tif"), source_root=src, dest_root=src
    )


def test_sparsify_roundtrip():
    img = np.zeros((32, 32), dtype=np.float32)
    img[3, 5] = 0.4
    img[10, 11] = 0.9
    sp = sparsify(img)
    assert sp.y.size == 2
    back = densify(sp, (32, 32))
    np.testing.assert_allclose(back, img)


def test_forbidden_and_dest_mapping(tmp_path, monkeypatch):
    protected = tmp_path / "protected"
    (protected / "DATA" / "ChanA").mkdir(parents=True)
    dest = tmp_path / "collected"
    monkeypatch.setattr(
        "batch_defringe.v4_export.WRITE_FORBIDDEN_ROOTS",
        (protected,),
    )
    tif = protected / "DATA" / "ChanA" / "ChanA_stk.tif"
    tif.write_bytes(b"x")
    assert forbidden_root_for(tif) == protected.resolve()
    try:
        assert_writable(protected / "DATA" / "ChanA" / "defringe_v4")
        raise AssertionError("should refuse")
    except RuntimeError:
        pass
    out = v4_dest_dir(tif, protected, dest)
    assert out == dest.resolve() / "DATA" / "ChanA" / "defringe_v4"
    assert forbidden_root_for(out) is None


def test_copy_xml(tmp_path):
    src_root = tmp_path / "src"
    trial = src_root / "mouse" / "ex01"
    trial.mkdir(parents=True)
    xml = trial / "Experiment.xml"
    xml.write_text(XML, encoding="utf-8")
    dest = tmp_path / "dest"
    copied = copy_experiment_xml(xml, src_root, dest, data_dir=trial / "DATA")
    assert copied is not None
    assert copied.is_file()
    assert copied.read_text(encoding="utf-8") == XML
    assert xml.is_file()


def test_grow_without_images():
    rec = grow_frame(_vstripes(), role="live", keep_images=False)
    assert rec["history"]
    assert rec["history"][0]["applied_sparse"].y is not None
    assert rec["raw"] is None
    assert "applied" not in rec["history"][0]


def test_process_stack_writes_dest_only(tmp_path):
    src_root = tmp_path / "ECF1"
    trial = src_root / "F1" / "ex01"
    chan = trial / "DATA" / "ChanA"
    chan.mkdir(parents=True)
    (trial / "Experiment.xml").write_text(XML, encoding="utf-8")
    stack = np.stack([_vstripes().astype(np.float32) for _ in range(3)])
    tif = chan / "ChanA_stk.tif"
    tifffile.imwrite(tif, stack, photometric="minisblack")
    dest = tmp_path / "CollectedData"
    job = job_for_stack(tif, root=src_root)
    result = process_stack_v4(
        tif,
        channel="ChanA",
        computer=job.computer,
        fingerprint=job.fingerprint,
        recording_date=job.date_utc,
        batch_root=src_root,
        source_root=src_root,
        dest_root=dest,
        xml_path=job.xml_path,
        data_dir=job.data_dir,
        skip_existing=False,
        slim_pdf=True,
        write_removed=False,
        write_means=False,
    )
    assert result.status == "ok"
    out = dest / "F1" / "ex01" / "DATA" / "ChanA" / "defringe_v4"
    assert result.out_dir == out.resolve() or result.out_dir == out
    assert (out / "ChanA_stk_defringed_v4.tif").is_file()
    assert (out / "per_frame.csv").is_file()
    assert (out / "families.json").is_file()
    assert (out / "mask_recipe.json").is_file()
    assert (out / "mask_patterns.npz").is_file()
    assert (out / "overview.pdf").is_file()
    assert (dest / "F1" / "ex01" / "Experiment.xml").is_file()
    assert not (chan / "defringe_v4").exists()
    assert not list(out.glob("*_removed_v4.tif"))
    assert not list(out.glob("mean_*.tif"))
    header = (out / "per_frame.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "removed_mean_abs" in header
    assert "n_history" in header
    loaded = load_mask_patterns(out / "mask_patterns.npz")
    assert loaded["n_frames"] == 3
    m0 = densify(slice_sparse(loaded["final_ptr"], loaded["final"], 0), (H, W))
    assert m0.shape == (H, W)
    recipe = (out / "mask_recipe.json").read_text(encoding="utf-8")
    assert "history" in recipe


def test_save_load_history_masks(tmp_path):
    a = np.zeros((8, 8), dtype=np.float32)
    a[1, 2] = 0.5
    b = np.zeros((8, 8), dtype=np.float32)
    b[3, 4] = 0.7
    path = tmp_path / "mask_patterns.npz"
    save_mask_patterns(
        path,
        shape_nwh=(2, 8, 8),
        finals=[sparsify(a), sparsify(b)],
        history_masks=[sparsify(a), sparsify(b)],
        history_frame=[0, 1],
        history_i=[0, 0],
        history_kept=[1, 0],
        history_alpha=[0.28, 0.55],
        peaks=[(0, "fx", 8.0, sparsify(a))],
        meta={"source_tif": "x"},
    )
    loaded = load_mask_patterns(path)
    assert int(loaded["history_kept"][1]) == 0
    np.testing.assert_allclose(
        densify(slice_sparse(loaded["final_ptr"], loaded["final"], 1), (8, 8)),
        b,
    )


if __name__ == "__main__":
    import tempfile

    test_sparsify_roundtrip()
    test_grow_without_images()
    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path as P

        class M:
            @staticmethod
            def setattr(name, val):
                import batch_defringe.v4_export as m

                m.WRITE_FORBIDDEN_ROOTS = val

        test_save_load_history_masks(P(d))
        test_copy_xml(P(d) / "xml")
        test_process_stack_writes_dest_only(P(d) / "stack")
    print("ok")
