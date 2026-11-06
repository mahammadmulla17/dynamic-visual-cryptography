"""Seven-stage flow orchestration for the prism-inspired visual cryptography demo."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import secrets
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image, ImageOps

from dvc import metrics
from prism_flow import (
    WAVELENGTHS_NM,
    classify_array,
    decrypt_share,
    encrypt_share,
    rgb_to_spectral,
)

DEFAULT_WORKDIR = Path(__file__).resolve().parent / "runs"
LOG_FILE = "sessions.jsonl"
WORKFLOW = "prism-flow/v1"
SHARE_IDS = ("share-A", "share-B")

FLOW_PHASES = [
    ("input", "1. Input image"),
    ("spectral", "2. Prism-inspired preprocessing"),
    ("split", "3. XOR share split"),
    ("encrypt", "4. AES-256-GCM"),
    ("channel", "5. Secure channel"),
    ("checks", "6. Receiver checks"),
    ("reconstruct", "7. XOR reconstruction"),
]

FAULT_KINDS = ["none", "drop", "delay", "flip", "crop", "substitute", "replay"]
FAULT_DESCRIPTIONS = {
    "none": "no fault",
    "flip": "flip a ciphertext bit",
    "tamper": "flip a ciphertext bit",
    "drop": "packet never arrives",
    "delay": "packet arrives late but intact",
    "crop": "truncate the ciphertext",
    "substitute": "swap in the other share's packet",
    "replay": "reuse a stale transfer id",
}


@dataclass
class FlowConfig:
    k: int = 2
    n: int = 2
    recovery: str = "xor"
    faults: dict[int, str] = field(default_factory=dict)
    persistent_faults: bool = False
    max_retries: int = 1
    store_seed: bool = True
    compute_secrecy: bool = True
    log: bool = True
    seed: int = 0

    def to_dict(self) -> dict:
        data = dict(self.__dict__)
        data["faults"] = {str(key): value for key, value in self.faults.items()}
        return data


@dataclass
class SecretImage:
    array: np.ndarray
    mode: str
    source_mode: str
    width: int
    height: int
    channels: int
    orientation_applied: bool = False
    alpha_flattened: bool = False

    def digest(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.array).tobytes()).hexdigest()


@dataclass
class Share:
    index: int
    array: np.ndarray
    scheme: str = "xor-2"
    tag: str = ""


@dataclass
class Reconstruction:
    display: np.ndarray
    raw: np.ndarray
    used_indices: list[int]


@dataclass
class Check:
    index: int
    status: str
    detail: str


@dataclass
class VerificationReport:
    checks: list[Check]
    admitted: list[int]
    k: int
    n: int


@dataclass
class FlowKey:
    nonce: bytes
    seed: int
    timestamp: float

    def seed_commitment(self) -> str:
        return hashlib.sha256(str(self.seed).encode("utf-8")).hexdigest()


@dataclass
class FlowSessionResult:
    session_id: str
    config: FlowConfig
    status: str = "pending"
    error: dict | None = None
    secret: SecretImage | None = None
    key: FlowKey | None = None
    shares: list[Share] = field(default_factory=list)
    manifest: dict | None = None
    receipts: list[dict] = field(default_factory=list)
    attempts: list[VerificationReport] = field(default_factory=list)
    reconstruction: Reconstruction | None = None
    below_threshold: np.ndarray | None = None
    scores: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)
    packets: list[dict] = field(default_factory=list)
    gcm_checks: list[dict] = field(default_factory=list)
    cnn_checks: list[dict | None] = field(default_factory=list)
    bands: np.ndarray | None = None
    transfer_id: str = ""
    packet_bytes: list[int] = field(default_factory=list)
    source_digest: str = ""
    seed: int = 0
    record: dict | None = None

    @property
    def ok(self) -> bool:
        return self.status == "success"

    @property
    def verification(self) -> VerificationReport | None:
        return self.attempts[-1] if self.attempts else None

    @property
    def source(self) -> np.ndarray | None:
        return self.secret.array if self.secret is not None else None

    @property
    def share_a(self) -> np.ndarray | None:
        return self.shares[0].array if len(self.shares) > 0 else None

    @property
    def share_b(self) -> np.ndarray | None:
        return self.shares[1].array if len(self.shares) > 1 else None

    @property
    def reconstructed(self) -> np.ndarray | None:
        return self.reconstruction.display if self.reconstruction is not None else None

    @property
    def blocked(self) -> bool:
        return self.status != "success"


def _load_source(source) -> tuple[np.ndarray, str, str, int, int, int, bool, bool]:
    orientation_applied = False
    alpha_flattened = False

    if isinstance(source, np.ndarray):
        array = np.asarray(source)
        source_mode = "RGB" if array.ndim == 3 and array.shape[-1] >= 3 else "L"
        image = Image.fromarray(array.astype(np.uint8))
    else:
        if isinstance(source, (bytes, bytearray)):
            with Image.open(BytesIO(source)) as opened:
                image = opened.copy()
        else:
            with Image.open(source) as opened:
                image = opened.copy()
        source_mode = image.mode
        exif = image.getexif()
        orientation = exif.get(274, 1) if exif else 1
        orientation_applied = orientation not in (1, None)
        image = ImageOps.exif_transpose(image)

    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        image = background
        alpha_flattened = True
    else:
        image = image.convert("RGB")

    if image.width < 8 or image.height < 8:
        raise ValueError("Input image is smaller than 8 pixels on a side")
    if max(image.size) > 2048:
        scale = 2048 / max(image.size)
        image = image.resize((max(8, round(image.width * scale)), max(8, round(image.height * scale))), Image.Resampling.LANCZOS)

    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    array = np.ascontiguousarray(array[..., :3], dtype=np.uint8)
    if source_mode == "1":
        mode = "binary"
    elif source_mode in ("L", "LA", "I", "F"):
        mode = "gray"
    else:
        mode = "color"
    return (
        array,
        mode,
        source_mode,
        int(array.shape[1]),
        int(array.shape[0]),
        int(array.shape[2]),
        orientation_applied,
        alpha_flattened,
    )


def _apply_fault(packet: dict, kind: str, packets: list[dict], index: int,
                 share_arrays: list[np.ndarray], master_key: bytes) -> dict | None:
    if kind == "drop":
        return None
    if kind == "none" or kind == "delay":
        return copy.deepcopy(packet)
    if kind == "substitute":
        return copy.deepcopy(packets[1 - index])
    altered = copy.deepcopy(packet)
    if kind in ("flip", "tamper"):
        data = bytearray(altered["ciphertext"])
        data[len(data) // 2] ^= 0x01
        altered["ciphertext"] = bytes(data)
        return altered
    if kind == "crop":
        altered["ciphertext"] = altered["ciphertext"][:-8]
        return altered
    if kind == "replay":
        return encrypt_share(
            share_arrays[index], master_key, packet["share_id"], secrets.token_hex(16)
        )
    return altered


def _transmit_and_verify(packets: list[dict], config: FlowConfig, master_key: bytes,
                         transfer_id: str, share_arrays: list[np.ndarray]):
    attempts: list[VerificationReport] = []
    receipts: list[dict] = []
    arrays: list[np.ndarray | None] = [None, None]

    for attempt_index in range(int(config.max_retries) + 1):
        apply_fault = attempt_index == 0 or config.persistent_faults
        checks: list[Check] = []
        arrays = [None, None]

        for index, expected_id in enumerate(SHARE_IDS):
            kind = config.faults.get(index + 1, "none") if apply_fault else "none"
            wire = _apply_fault(packets[index], kind, packets, index, share_arrays, master_key)
            if wire is None:
                checks.append(Check(index + 1, "MISSING", "packet did not arrive"))
                receipts.append({
                    "index": index + 1, "channel": index + 1, "attempt": attempt_index + 1,
                    "status": "dropped", "fault": kind, "bytes_written": 0,
                })
                continue
            if wire["share_id"] != expected_id:
                checks.append(Check(index + 1, "TAMPER", "share identity mismatch"))
                receipts.append({
                    "index": index + 1, "channel": index + 1, "attempt": attempt_index + 1,
                    "status": "rejected", "fault": kind, "bytes_written": len(wire["ciphertext"]),
                })
                continue
            try:
                array = decrypt_share(wire, master_key, transfer_id)
                arrays[index] = array
                checks.append(Check(index + 1, "VALID", "GCM tag verified"))
                receipts.append({
                    "index": index + 1, "channel": index + 1, "attempt": attempt_index + 1,
                    "status": "delivered late" if kind == "delay" else "delivered",
                    "fault": kind, "bytes_written": len(wire["ciphertext"]),
                })
            except ValueError as exc:
                status = "STALE" if "different transfer" in str(exc) else "TAMPER"
                checks.append(Check(index + 1, status, str(exc)))
                receipts.append({
                    "index": index + 1, "channel": index + 1, "attempt": attempt_index + 1,
                    "status": "rejected", "fault": kind, "bytes_written": len(wire["ciphertext"]),
                })

        attempts.append(VerificationReport(checks, [c.index for c in checks if c.status == "VALID"], 2, 2))
        if all(check.status == "VALID" for check in checks):
            break

    final = attempts[-1]
    gcm_checks = [
        {
            "share": SHARE_IDS[index],
            "verified": check.status == "VALID",
            "detail": check.detail,
            "status": check.status,
            "nonce": packets[index]["nonce"].hex(),
            "salt": packets[index]["salt"].hex(),
        }
        for index, check in enumerate(final.checks)
    ]
    return attempts, receipts, gcm_checks, arrays


def _score(result: FlowSessionResult, config: FlowConfig) -> dict:
    source = result.secret.array
    exact = result.reconstructed is not None and bool(np.array_equal(source, result.reconstructed))
    if result.reconstructed is None:
        psnr_db = 0.0
        ssim_value = 0.0
        mae = 0.0
    else:
        psnr_db = metrics.psnr(source, result.reconstructed)
        ssim_value = metrics.ssim(source, result.reconstructed)
        mae = float(np.mean(np.abs(source.astype(np.int16) - result.reconstructed.astype(np.int16))))

    per_share = []
    for share in result.shares:
        per_share.append({
            "index": share.index,
            "entropy_bits": metrics.shannon_entropy(share.array),
            "corr_to_secret": metrics.pearson(share.array, source),
            "adjacent_corr": metrics.adjacent_correlation(share.array),
        })
    min_entropy = min(item["entropy_bits"] for item in per_share) if per_share else 0.0
    max_corr = max(abs(item["corr_to_secret"]) for item in per_share) if per_share else 1.0
    secrecy = {"per_share": per_share, "min_share_entropy": min_entropy, "max_abs_share_corr": max_corr}

    if config.compute_secrecy and result.share_a is not None:
        rng = np.random.default_rng(result.seed + 1)
        other = rng.integers(0, 256, size=result.share_a.shape, dtype=np.uint8)
        secrecy["cross_session"] = {
            "npcr_min": metrics.npcr(result.share_a, other),
            "uaci_mean": metrics.uaci(result.share_a, other),
            "max_abs_corr": abs(metrics.pearson(result.share_a, other)),
        }

    gcm_passed = sum(1 for check in result.gcm_checks if check["verified"])
    cnn_passed = sum(1 for check in result.cnn_checks if check and check["label"] == "genuine")
    targets = [
        {
            "metric": "Exact reconstruction",
            "value": "yes" if exact else "no",
            "target": "bit-exact",
            "passed": exact,
            "note": "XOR of Share A and Share B",
        },
        {
            "metric": "PSNR",
            "value": psnr_db,
            "target": "> 30 dB",
            "passed": exact or psnr_db > 30.0,
            "note": "recovered vs source",
        },
        {
            "metric": "MAE",
            "value": mae,
            "target": "0",
            "passed": mae == 0.0,
            "note": "mean absolute pixel error",
        },
        {
            "metric": "AES-GCM tags verified",
            "value": f"{gcm_passed}/2",
            "target": "2/2",
            "passed": gcm_passed == 2,
            "note": "authenticated before decryption",
        },
        {
            "metric": "CNN shares genuine",
            "value": f"{cnn_passed}/2",
            "target": "2/2",
            "passed": cnn_passed == 2,
            "note": "secondary demo check",
        },
        {
            "metric": "Min share entropy",
            "value": min_entropy,
            "target": "> 7.9 bits",
            "passed": min_entropy > 7.9,
            "note": "per-share randomness",
        },
        {
            "metric": "Max share-to-secret |corr|",
            "value": max_corr,
            "target": "< 0.05",
            "passed": max_corr < 0.05,
            "note": "statistical independence",
        },
    ]
    return {
        "fidelity": {"psnr_db": psnr_db, "ssim": ssim_value, "mae": mae},
        "secrecy": secrecy,
        "targets": targets,
    }


def _append_log(result: FlowSessionResult, workdir: Path, label: str) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    inputs = workdir / "inputs"
    inputs.mkdir(exist_ok=True)
    input_path = inputs / f"{result.source_digest}.png"
    if not input_path.exists():
        Image.fromarray(result.source).save(input_path)

    record = {
        "workflow": WORKFLOW,
        "session_id": result.session_id,
        "status": result.status,
        "error": result.error,
        "config": result.config.to_dict(),
        "image_label": label,
        "image_mode": result.secret.mode if result.secret else "",
        "source_mode": result.secret.source_mode if result.secret else "",
        "width": result.secret.width if result.secret else 0,
        "height": result.secret.height if result.secret else 0,
        "source_digest": result.source_digest,
        "seed": result.seed if result.config.store_seed else None,
        "seed_commitment": result.key.seed_commitment() if result.key else None,
        "transfer_id": result.transfer_id,
        "tags": [share.tag for share in result.shares],
        "recovered_digest": (
            hashlib.sha256(np.ascontiguousarray(result.reconstructed).tobytes()).hexdigest()
            if result.reconstructed is not None else None
        ),
        "packet_bytes": result.packet_bytes,
        "gcm": [
            {"share": check["share"], "verified": check["verified"], "status": check["status"]}
            for check in result.gcm_checks
        ],
        "cnn": [
            {"label": check["label"]} if check is not None else None
            for check in result.cnn_checks
        ],
        "metrics": {
            "exact": result.scores.get("fidelity", {}).get("mae") == 0.0,
            "psnr_db": result.scores.get("fidelity", {}).get("psnr_db"),
            "gcm_passed": sum(1 for check in result.gcm_checks if check["verified"]),
        },
    }
    with open(workdir / LOG_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    result.record = record


def read_flow_log(workdir: Path = DEFAULT_WORKDIR) -> list[dict]:
    path = workdir / LOG_FILE
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("workflow") == WORKFLOW:
            records.append(record)
    return records


def replay_flow(session_id: str, model, workdir: Path = DEFAULT_WORKDIR) -> dict:
    records = read_flow_log(workdir)
    record = next((item for item in records if item.get("session_id") == session_id), None)
    if record is None:
        raise ValueError("Session not found in the flow log")
    if record.get("seed") is None:
        raise ValueError("Session was logged without a seed, so it cannot be replayed")
    input_path = workdir / "inputs" / f"{record['source_digest']}.png"
    if not input_path.exists():
        raise ValueError("The logged input image is missing")

    config_data = record.get("config", {})
    config = FlowConfig(
        k=2,
        n=2,
        recovery="xor",
        faults={int(key): value for key, value in config_data.get("faults", {}).items()},
        persistent_faults=bool(config_data.get("persistent_faults", False)),
        max_retries=int(config_data.get("max_retries", 1)),
        store_seed=True,
        compute_secrecy=False,
        log=False,
        seed=int(record["seed"]),
    )
    start = time.perf_counter()
    outcome = run_flow_session(
        input_path,
        config,
        model,
        workdir,
        label=record.get("image_label", ""),
        session_id=session_id,
        log=False,
    )
    tag_matches = sum(
        1 for tag, expected in zip((share.tag for share in outcome.shares), record.get("tags", []))
        if tag == expected
    )
    recovered_digest = (
        hashlib.sha256(np.ascontiguousarray(outcome.reconstructed).tobytes()).hexdigest()
        if outcome.reconstructed is not None else None
    )
    output_match = recovered_digest == record.get("recovered_digest")
    return {
        "bit_exact": tag_matches == len(record.get("tags", [])) and output_match,
        "tag_matches": tag_matches,
        "tag_count": len(record.get("tags", [])),
        "output_match": output_match,
        "reconstruction": outcome.reconstruction,
        "replay_ms": (time.perf_counter() - start) * 1000.0,
    }


def run_flow_session(source, config: FlowConfig, model, workdir: Path = DEFAULT_WORKDIR,
                     label: str = "", session_id: str | None = None,
                     log: bool = True) -> FlowSessionResult:
    session_id = session_id or secrets.token_hex(8)
    result = FlowSessionResult(session_id=session_id, config=config)
    result.seed = int(config.seed) if config.seed else secrets.randbits(63)
    current_phase = "input"

    try:
        start = time.perf_counter()
        result.gates[current_phase] = "running"
        array, mode, source_mode, width, height, channels, orientation, alpha = _load_source(source)
        result.secret = SecretImage(
            array=array,
            mode=mode,
            source_mode=source_mode,
            width=width,
            height=height,
            channels=channels,
            orientation_applied=orientation,
            alpha_flattened=alpha,
        )
        result.source_digest = result.secret.digest()
        result.timings["input_ms"] = (time.perf_counter() - start) * 1000.0
        result.gates[current_phase] = "passed"

        current_phase = "spectral"
        start = time.perf_counter()
        result.gates[current_phase] = "running"
        result.bands = rgb_to_spectral(array)
        result.timings["spectral_ms"] = (time.perf_counter() - start) * 1000.0
        result.gates[current_phase] = "passed"

        current_phase = "split"
        start = time.perf_counter()
        result.gates[current_phase] = "running"
        rng = np.random.default_rng(result.seed)
        share_a = rng.integers(0, 256, size=array.shape, dtype=np.uint8)
        share_b = np.bitwise_xor(array, share_a)
        tags = [
            hashlib.sha256(np.ascontiguousarray(share_a).tobytes()).hexdigest(),
            hashlib.sha256(np.ascontiguousarray(share_b).tobytes()).hexdigest(),
        ]
        result.shares = [
            Share(1, share_a, "xor-2", tags[0]),
            Share(2, share_b, "xor-2", tags[1]),
        ]
        result.timings["split_ms"] = (time.perf_counter() - start) * 1000.0
        result.gates[current_phase] = "passed"

        current_phase = "encrypt"
        start = time.perf_counter()
        result.gates[current_phase] = "running"
        master_key = AESGCM.generate_key(bit_length=256)
        result.transfer_id = secrets.token_hex(16)
        result.packets = [
            encrypt_share(share_a, master_key, SHARE_IDS[0], result.transfer_id),
            encrypt_share(share_b, master_key, SHARE_IDS[1], result.transfer_id),
        ]
        result.packet_bytes = [len(packet["ciphertext"]) for packet in result.packets]
        result.key = FlowKey(result.packets[0]["nonce"], result.seed, time.time())
        result.manifest = {
            "workflow": WORKFLOW,
            "session_id": session_id,
            "algorithm": "prism-xor-aes256gcm-cnn/v1",
            "k": 2,
            "n": 2,
            "width": width,
            "height": height,
            "transfer_id": result.transfer_id,
            "tags": [
                {"index": share.index, "tag": share.tag} for share in result.shares
            ],
        }
        result.timings["encrypt_ms"] = (time.perf_counter() - start) * 1000.0
        result.gates[current_phase] = "passed"

        current_phase = "channel"
        start = time.perf_counter()
        result.gates[current_phase] = "running"
        attempts, receipts, gcm_checks, received = _transmit_and_verify(
            result.packets, config, master_key, result.transfer_id, [share_a, share_b]
        )
        result.attempts = attempts
        result.receipts = receipts
        result.gcm_checks = gcm_checks
        result.below_threshold = share_a
        delivered = all(
            receipt["status"] != "dropped" for receipt in receipts if receipt["attempt"] == len(attempts)
        )
        result.gates[current_phase] = "passed" if delivered else "failed"
        result.timings["channel_ms"] = (time.perf_counter() - start) * 1000.0

        current_phase = "checks"
        start = time.perf_counter()
        result.gates[current_phase] = "running"
        result.cnn_checks = []
        for array_value in received:
            if array_value is None:
                result.cnn_checks.append(None)
            else:
                prediction, probabilities = classify_array(model, array_value)
                result.cnn_checks.append({
                    "label": prediction,
                    "probabilities": dict(zip(("genuine", "noisy", "tampered"), probabilities)),
                })
        gcm_ok = all(check["verified"] for check in gcm_checks)
        cnn_ok = all(
            check is not None and check["label"] == "genuine" for check in result.cnn_checks
        )
        result.timings["checks_ms"] = (time.perf_counter() - start) * 1000.0
        result.gates[current_phase] = "passed" if gcm_ok and cnn_ok else "failed"

        current_phase = "reconstruct"
        start = time.perf_counter()
        result.gates[current_phase] = "running"
        if all(value is not None for value in received):
            reconstructed = np.bitwise_xor(received[0], received[1])
            result.reconstruction = Reconstruction(reconstructed, reconstructed, [1, 2])
            result.status = "success"
            result.gates[current_phase] = "passed"
        else:
            failed = next(check for check in gcm_checks if not check["verified"])
            result.status = "aborted"
            result.error = {
                "code": failed["status"],
                "phase": "6. Receiver checks",
                "message": f"{failed['share']} rejected: {failed['detail']}",
            }
            result.gates[current_phase] = "skipped"
        result.timings["reconstruct_ms"] = (time.perf_counter() - start) * 1000.0

        result.scores = _score(result, config)

        if log and config.log:
            _append_log(result, workdir, label)
        return result
    except Exception as exc:
        result.status = "aborted"
        result.error = {
            "code": "BAD_INPUT" if current_phase == "input" else "FLOW_ERROR",
            "phase": current_phase,
            "message": str(exc),
        }
        result.gates[current_phase] = "failed"
        if log and config.log and result.secret is not None:
            try:
                result.scores = _score(result, config)
                _append_log(result, workdir, label)
            except Exception:
                pass
        return result
