"""MCA demonstration: prism-inspired spectral preprocessing, XOR shares, AES-256-GCM, CNN."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import secrets
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from PIL import Image, ImageDraw, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LABELS = ("genuine", "noisy", "tampered")
WAVELENGTHS_NM = (400, 450, 500, 550, 600, 650, 700)
SPECTRAL_WEIGHTS = np.asarray(
    [
        [0.05, 0.02, 0.93],
        [0.04, 0.14, 0.82],
        [0.02, 0.50, 0.48],
        [0.04, 0.91, 0.05],
        [0.46, 0.52, 0.02],
        [0.86, 0.13, 0.01],
        [0.97, 0.03, 0.00],
    ],
    dtype=np.float32,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rgb_to_spectral(rgb: np.ndarray) -> np.ndarray:
    """Map uint8 HxWx3 RGB to normalized HxWx7 simulated bands."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected an RGB image with shape HxWx3")
    if rgb.dtype != np.uint8:
        raise ValueError("RGB input must use uint8 values")
    normalized = rgb.astype(np.float32) / 255.0
    bands = np.einsum("hwc,bc->hwb", normalized, SPECTRAL_WEIGHTS)
    return np.clip(bands, 0.0, 1.0).astype(np.float32)


def spectral_band_to_rgb(band: np.ndarray, wavelength_nm: int) -> np.ndarray:
    tint = {
        400: (86, 40, 255),
        450: (30, 90, 255),
        500: (20, 190, 210),
        550: (45, 205, 65),
        600: (235, 205, 40),
        650: (255, 120, 25),
        700: (255, 35, 45),
    }[wavelength_nm]
    gray = np.clip(band, 0.0, 1.0)[..., None]
    return np.clip(gray * np.asarray(tint, dtype=np.float32)[None, None, :], 0, 255).astype(np.uint8)


def synthetic_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def corrupt(rgb: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    out = rgb.copy()
    h, w, _ = out.shape
    if kind == "noisy":
        noise = rng.normal(0, rng.uniform(10, 38), size=out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if rng.random() < 0.65:
            channel = int(rng.integers(0, 3))
            out[..., channel] = 0
    elif kind == "tampered":
        for _ in range(int(rng.integers(1, 4))):
            bh = int(rng.integers(max(4, h // 6), max(5, h // 2)))
            bw = int(rng.integers(max(4, w // 6), max(5, w // 2)))
            y, x = int(rng.integers(0, h - bh + 1)), int(rng.integers(0, w - bw + 1))
            style = int(rng.integers(0, 4))
            if style == 0:
                out[y : y + bh, x : x + bw] = int(rng.integers(0, 256))
            elif style == 1:
                yy, xx = np.mgrid[0:bh, 0:bw]
                dots = ((yy // 2 + xx // 2) % 2).astype(np.uint8)
                color = rng.integers(0, 256, size=3)
                out[y : y + bh, x : x + bw] = dots[..., None] * color
            elif style == 2:
                patch = Image.fromarray(out[y : y + bh, x : x + bw])
                patch = patch.filter(ImageFilter.GaussianBlur(radius=4))
                out[y : y + bh, x : x + bw] = np.asarray(patch)
            else:
                stripe = rng.integers(0, 256, size=(1, bw, 3), dtype=np.uint8)
                out[y : y + bh, x : x + bw] = np.tile(stripe, (bh, 1, 1))
    return out


def make_dataset(out_dir: Path, samples_per_class: int = 240, size: int = 64, seed: int = 42) -> Path:
    if samples_per_class < 12 or size < 16:
        raise ValueError("Use at least 12 samples per class and image size 16")
    rng = np.random.default_rng(seed)
    samples, labels = [], []
    for label_index, label in enumerate(LABELS):
        for _ in range(samples_per_class):
            image = synthetic_mask(size, rng)
            if label != "genuine":
                image = corrupt(image, label, rng)
            samples.append(rgb_to_spectral(image).transpose(2, 0, 1))
            labels.append(label_index)
    samples = np.asarray(samples, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(len(labels))
    samples, labels = samples[order], labels[order]
    n = len(labels)
    a, b = int(n * 0.70), int(n * 0.85)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "shares.npz"
    np.savez_compressed(target, x=samples, y=labels, train_end=a, val_end=b, labels=np.asarray(LABELS))
    return target


class ShareCNN(nn.Module):
    def __init__(self, channels: int = 7, classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(64, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def load_data(path: Path):
    data = np.load(path, allow_pickle=False)
    x, y, train_end, val_end = data["x"], data["y"], int(data["train_end"]), int(data["val_end"])
    return (x[:train_end], y[:train_end]), (x[train_end:val_end], y[train_end:val_end]), (x[val_end:], y[val_end:])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        cm[int(actual), int(predicted)] += 1
    precision, recall, f1 = [], [], []
    for i in range(len(LABELS)):
        p = cm[i, i] / max(1, cm[:, i].sum())
        r = cm[i, i] / max(1, cm[i, :].sum())
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / max(1e-12, p + r))
    return {
        "accuracy": float(np.trace(cm) / max(1, cm.sum())),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix_rows_actual_cols_predicted": cm.tolist(),
    }


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        return logits.argmax(1).cpu().numpy()


def train_model(data_path: Path, model_path: Path, epochs: int = 8, seed: int = 42, log=print) -> dict:
    seed_everything(seed)
    (xtr, ytr), (xv, yv), (xt, yt) = load_data(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShareCNN().to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)), batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(yb)
        val_metrics = metrics(yv, predict(model, xv, device))
        log(
            f"epoch {epoch}/{epochs} loss={running / len(ytr):.4f} "
            f"val_accuracy={val_metrics['accuracy']:.3f} val_macro_f1={val_metrics['macro_f1']:.3f}"
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Training did not produce a model")
    model.load_state_dict(best_state)
    test_metrics = metrics(yt, predict(model, xt, device))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": LABELS,
            "wavelengths_nm": WAVELENGTHS_NM,
            "seed": seed,
        },
        model_path,
    )
    report = {"device": str(device), "test": test_metrics}
    return report


def load_model(model_path: Path) -> nn.Module:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ShareCNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def classify_array(model: nn.Module, rgb: np.ndarray) -> tuple[str, list[float]]:
    device = next(model.parameters()).device
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    if image.width >= 64 and image.height >= 64:
        left = (image.width - 64) // 2
        top = (image.height - 64) // 2
        image = image.crop((left, top, left + 64, top + 64))
    else:
        image = image.resize((64, 64), Image.Resampling.NEAREST)
    x = rgb_to_spectral(np.asarray(image)).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(x).to(device)), dim=1)[0].cpu().numpy()
    index = int(probabilities.argmax())
    return LABELS[index], probabilities.tolist()


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def demo_image(source: Path | np.ndarray | None, size: int = 64) -> np.ndarray:
    if source is not None:
        if isinstance(source, np.ndarray):
            im = Image.fromarray(source).convert("RGB")
        else:
            with Image.open(source) as opened:
                im = opened.convert("RGB")
        return np.asarray(im.resize((size, size), Image.Resampling.LANCZOS), dtype=np.uint8)
    im = Image.new("RGB", (size, size), "#203050")
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=10, fill="#1f2a44", outline="#f2c14e", width=2)
    draw.ellipse((8, 8, size - 8, size - 8), fill="#eea631", outline="white", width=2)
    draw.polygon([(size // 2, 15), (size - 15, size // 2), (size // 2, size - 15), (15, size // 2)], fill="#2978b8")
    draw.ellipse((size // 3, size // 3, 2 * size // 3, 2 * size // 3), fill="#ee5260")
    return np.asarray(im, dtype=np.uint8)


def _share_aad(packet: dict) -> bytes:
    fields = {name: packet[name] for name in ("protocol", "version", "transfer_id", "share_id", "shape", "dtype")}
    return json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _derive_share_key(master_key: bytes, salt: bytes, transfer_id: str, share_id: str) -> bytes:
    if len(master_key) != 32:
        raise ValueError("AES-256 master key must be exactly 32 bytes")
    info = f"DVC-AES-256-GCM/v2/{transfer_id}/{share_id}".encode("utf-8")
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=info).derive(master_key)


def encrypt_share(share: np.ndarray, master_key: bytes, share_id: str, transfer_id: str) -> dict:
    """Encrypt one share with a per-share HKDF key, fresh salt, and 96-bit GCM nonce."""
    if share.dtype != np.uint8 or share.ndim != 3 or share.shape[-1] != 3:
        raise ValueError("Share must be a uint8 HxWx3 RGB array")
    packet = {
        "protocol": "DVC",
        "version": 2,
        "transfer_id": transfer_id,
        "share_id": share_id,
        "shape": list(share.shape),
        "dtype": "uint8",
        "kdf": "HKDF-SHA256",
        "cipher": "AES-256-GCM",
        "salt": secrets.token_bytes(16),
        "nonce": secrets.token_bytes(12),
    }
    key = _derive_share_key(master_key, packet["salt"], transfer_id, share_id)
    packet["ciphertext"] = AESGCM(key).encrypt(packet["nonce"], share.tobytes(), _share_aad(packet))
    return packet


def decrypt_share(packet: dict, master_key: bytes, expected_transfer_id: str | None = None) -> np.ndarray:
    required = {
        "protocol",
        "version",
        "transfer_id",
        "share_id",
        "shape",
        "dtype",
        "kdf",
        "cipher",
        "salt",
        "nonce",
        "ciphertext",
    }
    if not isinstance(packet, dict) or not required.issubset(packet):
        raise ValueError("Malformed encrypted-share envelope")
    if packet["protocol"] != "DVC" or packet["version"] != 2 or packet["dtype"] != "uint8":
        raise ValueError("Unsupported encrypted-share protocol or version")
    if packet["kdf"] != "HKDF-SHA256" or packet["cipher"] != "AES-256-GCM":
        raise ValueError("Unsupported key derivation or cipher")
    if expected_transfer_id is not None and packet["transfer_id"] != expected_transfer_id:
        raise ValueError("Share belongs to a different transfer (possible replay or mix-up)")
    if len(packet["salt"]) != 16 or len(packet["nonce"]) != 12:
        raise ValueError("Invalid HKDF salt or AES-GCM nonce length")
    shape = packet["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or shape[2] != 3
        or any(not isinstance(v, int) or v < 1 or v > 4096 for v in shape)
    ):
        raise ValueError("Invalid share dimensions")
    key = _derive_share_key(master_key, packet["salt"], packet["transfer_id"], packet["share_id"])
    try:
        raw = AESGCM(key).decrypt(packet["nonce"], packet["ciphertext"], _share_aad(packet))
    except InvalidTag as exc:
        raise ValueError("AES-GCM authentication failed: wrong key or altered share/metadata") from exc
    expected = int(np.prod(shape))
    if len(raw) != expected:
        raise ValueError("Authenticated payload has an invalid length")
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()


@dataclass
class FlowResult:
    source: np.ndarray
    bands: np.ndarray
    share_a: np.ndarray
    share_b: np.ndarray
    packets: list[dict]
    received: list[np.ndarray | None]
    gcm_checks: list[dict]
    cnn_checks: list[dict | None]
    reconstructed: np.ndarray | None
    source_digest: str
    transfer_id: str
    packet_bytes: list[int]
    psnr_db: float
    mae: float
    exact: bool
    blocked: bool = False
    extras: dict = field(default_factory=dict)


def run_flow(
    image_path: Path | np.ndarray | None,
    model: nn.Module,
    size: int = 64,
    seed: int = 42,
    tamper: bool = False,
    master_key: bytes | None = None,
) -> FlowResult:
    source = demo_image(image_path, size)
    bands = rgb_to_spectral(source)
    rng = np.random.default_rng(seed)
    share_a = synthetic_mask(size, rng)
    share_b = np.bitwise_xor(source, share_a)

    master_key = master_key or AESGCM.generate_key(bit_length=256)
    transfer_id = secrets.token_hex(16)
    packets = [
        encrypt_share(share_a, master_key, "share-A", transfer_id),
        encrypt_share(share_b, master_key, "share-B", transfer_id),
    ]

    wire = copy.deepcopy(packets)
    if tamper:
        altered = bytearray(wire[1]["ciphertext"])
        altered[len(altered) // 2] ^= 0x01
        wire[1]["ciphertext"] = bytes(altered)

    received: list[np.ndarray | None] = []
    gcm_checks: list[dict] = []
    for index, (expected, packet) in enumerate(zip(("share-A", "share-B"), wire), start=1):
        try:
            array = decrypt_share(packet, master_key, transfer_id)
            received.append(array)
            gcm_checks.append(
                {
                    "share": expected,
                    "verified": True,
                    "detail": "GCM tag verified",
                    "nonce": packet["nonce"].hex(),
                    "salt": packet["salt"].hex(),
                }
            )
        except ValueError as exc:
            received.append(None)
            gcm_checks.append(
                {
                    "share": expected,
                    "verified": False,
                    "detail": str(exc),
                    "nonce": packet["nonce"].hex(),
                    "salt": packet["salt"].hex(),
                }
            )

    cnn_checks: list[dict | None] = []
    for array in received:
        if array is None:
            cnn_checks.append(None)
        else:
            label, probabilities = classify_array(model, array)
            cnn_checks.append({"label": label, "probabilities": dict(zip(LABELS, probabilities))})

    blocked = any(array is None for array in received)
    reconstructed = None if blocked else np.bitwise_xor(received[0], received[1])
    if reconstructed is None:
        psnr_db, mae, exact = 0.0, 0.0, False
    else:
        mse = float(np.mean((source.astype(np.float32) - reconstructed.astype(np.float32)) ** 2))
        psnr_db = float("inf") if mse == 0 else 10 * math.log10((255.0**2) / mse)
        mae = float(np.mean(np.abs(source.astype(np.int16) - reconstructed.astype(np.int16))))
        exact = bool(np.array_equal(source, reconstructed))

    return FlowResult(
        source=source,
        bands=bands,
        share_a=share_a,
        share_b=share_b,
        packets=packets,
        received=received,
        gcm_checks=gcm_checks,
        cnn_checks=cnn_checks,
        reconstructed=reconstructed,
        source_digest=hashlib.sha256(source.tobytes()).hexdigest(),
        transfer_id=transfer_id,
        packet_bytes=[len(packet["ciphertext"]) for packet in packets],
        psnr_db=psnr_db,
        mae=mae,
        exact=exact,
        blocked=blocked,
    )


def run_demo(args) -> None:
    work = Path(args.workdir)
    data_path = work / "dataset" / "shares.npz"
    model_path = work / "share_cnn.pt"
    make_dataset(data_path.parent, args.samples_per_class, args.size, args.seed)
    train_model(data_path, model_path, args.epochs, args.seed)
    model = load_model(model_path)
    result = run_flow(args.image, model, size=args.size, seed=args.seed)
    save_rgb(result.source, work / "source.png")
    save_rgb(result.share_a, work / "share_1.png")
    save_rgb(result.share_b, work / "share_2.png")
    save_rgb(result.reconstructed, work / "reconstructed.png")
    report = {
        "spectral_bands_nm_simulated": WAVELENGTHS_NM,
        "classification": [
            {"share": i + 1, "predicted": check["label"], "probabilities": check["probabilities"]}
            for i, check in enumerate(result.cnn_checks)
            if check
        ],
        "reconstruction": {"mae": result.mae, "psnr_db": result.psnr_db, "exact": result.exact},
        "encryption": {
            "cipher": "AES-256-GCM",
            "kdf": "HKDF-SHA256",
            "nonce_bits": 96,
            "salt_bits": 128,
            "status": "both shares authenticated and decrypted successfully"
            if not result.blocked
            else "receiver blocked: authentication failed",
        },
        "key_note": "ephemeral demo master key; not saved",
    }
    (work / "demo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Artifacts: {work.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate-data")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--samples-per-class", type=int, default=240)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("train")
    p.add_argument("--data", type=Path, default=Path("data/shares.npz"))
    p.add_argument("--model", type=Path, default=Path("share_cnn.pt"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("evaluate")
    p.add_argument("--data", type=Path, default=Path("data/shares.npz"))
    p.add_argument("--model", type=Path, default=Path("share_cnn.pt"))
    p = sub.add_parser("infer")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p = sub.add_parser("demo")
    p.add_argument("--image", type=Path)
    p.add_argument("--workdir", type=Path, default=Path("demo_output"))
    p.add_argument("--samples-per-class", type=int, default=120)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "generate-data":
        make_dataset(args.out, args.samples_per_class, args.size, args.seed)
    elif args.command == "train":
        train_model(args.data, args.model, args.epochs, args.seed)
    elif args.command == "evaluate":
        (_, _), (_, _), (xt, yt) = load_data(args.data)
        model = load_model(args.model)
        print(json.dumps(metrics(yt, predict(model, xt, torch.device("cpu"))), indent=2))
    elif args.command == "infer":
        model = load_model(args.model)
        with Image.open(args.image) as im:
            rgb = np.asarray(im.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS), dtype=np.uint8)
        label, probs = classify_array(model, rgb)
        print(json.dumps({"prediction": label, "probabilities": dict(zip(LABELS, probs))}, indent=2))
    else:
        run_demo(args)


if __name__ == "__main__":
    main()
