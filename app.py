from __future__ import annotations

import base64
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from diagram import build_flow_svg
from dvc.errors import EXPLANATIONS
from flow_session import (
    DEFAULT_WORKDIR,
    FAULT_DESCRIPTIONS,
    FAULT_KINDS,
    FLOW_PHASES,
    FlowConfig,
    read_flow_log,
    replay_flow,
    run_flow_session,
)
from prism_flow import WAVELENGTHS_NM, load_model, make_dataset, spectral_band_to_rgb, train_model

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
WORKDIR = DEFAULT_WORKDIR
MODEL_PATH = ROOT / "share_cnn.pt"
DATA_PATH = ROOT / "data" / "shares.npz"

st.set_page_config(page_title="Dynamic Visual Cryptography", layout="wide")

GATE_ICON = {"passed": "✅", "failed": "❌", "skipped": "⏭️", "running": "⏳"}
STATUS_ICON = {"VALID": "✅", "MISSING": "📭", "TAMPER": "🛑", "STALE": "⌛", "PARAM_MISMATCH": "⚠️"}


def to_png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def fmt(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isinf(value):
            return "∞ (lossless)"
        return f"{value:.4f}"
    if isinstance(value, str) and value in ("inf", "Infinity"):
        return "∞ (lossless)"
    return str(value)


def sample_files() -> list[Path]:
    if not SAMPLES.exists():
        return []
    return sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})


def diagram_html(result) -> str:
    encoded = base64.b64encode(build_flow_svg(result).encode("utf-8")).decode("ascii")
    return (
        f'<img src="data:image/svg+xml;base64,{encoded}" '
        'alt="Seven-stage dynamic visual cryptography flow" '
        'style="width:100%;height:auto;display:block;" />'
    )


def report_payload(result, label: str) -> dict:
    return {
        "workflow": "prism-flow/v1",
        "session_id": result.session_id,
        "label": label,
        "status": result.status,
        "error": result.error,
        "config": {
            "k": result.config.k, "n": result.config.n, "recovery": result.config.recovery,
            "faults": result.config.faults, "persistent_faults": result.config.persistent_faults,
            "max_retries": result.config.max_retries, "store_seed": result.config.store_seed,
            "compute_secrecy": result.config.compute_secrecy,
        },
        "source": {
            "mode": result.secret.mode if result.secret else "",
            "source_mode": result.secret.source_mode if result.secret else "",
            "width": result.secret.width if result.secret else 0,
            "height": result.secret.height if result.secret else 0,
            "channels": result.secret.channels if result.secret else 0,
            "sha256": result.source_digest,
        },
        "spectral_bands_nm_simulated": list(WAVELENGTHS_NM),
        "share_split": {
            "scheme": "xor-2",
            "shares": [{"index": share.index, "tag": share.tag} for share in result.shares],
        },
        "encryption": {
            "cipher": "AES-256-GCM",
            "kdf": "HKDF-SHA256",
            "transfer_id": result.transfer_id,
            "packet_bytes": result.packet_bytes,
        },
        "channel": {
            "receipts": result.receipts,
            "attempts": len(result.attempts),
        },
        "receiver_checks": {
            "gcm": [
                {"share": check["share"], "verified": check["verified"], "status": check["status"]}
                for check in result.gcm_checks
            ],
            "cnn": [
                {"label": check["label"]} if check is not None else None for check in result.cnn_checks
            ],
        },
        "reconstruction": {
            "exact": result.config.recovery == "xor" and result.ok,
            "psnr_db": result.scores.get("fidelity", {}).get("psnr_db"),
            "mae": result.scores.get("fidelity", {}).get("mae"),
        },
        "metrics": result.scores.get("targets", []),
        "timings_ms": result.timings,
    }


@st.cache_resource(show_spinner=False)
def get_model():
    if not MODEL_PATH.exists():
        if not DATA_PATH.exists():
            make_dataset(DATA_PATH.parent, samples_per_class=400, size=64, seed=42)
        train_model(DATA_PATH, MODEL_PATH, epochs=14, seed=42)
    return load_model(MODEL_PATH)


st.sidebar.title("Session setup")

source_kind = st.sidebar.radio("Secret image", ["Sample", "Upload"], horizontal=True)
image_source = None
image_label = ""
if source_kind == "Sample":
    files = sample_files()
    if files:
        choice = st.sidebar.selectbox("Sample image", files, format_func=lambda p: p.stem)
        image_source = choice
        image_label = choice.stem
    else:
        st.sidebar.warning("No images in samples/.")
else:
    upload = st.sidebar.file_uploader("PNG, JPEG, TIFF or BMP (max 2048 x 2048)",
                                      type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    if upload is not None:
        image_source = upload.getvalue()
        image_label = upload.name

preset = st.sidebar.selectbox("Threshold preset (fixed for this flow)", ["(2, 2) two-party escrow"])
k, n = 2, 2
st.sidebar.caption("This flow is a fixed 2-of-2 XOR split: Share A is fresh noise and Share B is required for "
                   "recovery. Other thresholds and OR stacking need a different share scheme.")

recovery_label = st.sidebar.radio(
    "Recovery mode (fixed for this flow)", ["XOR (digital, lossless)"],
    help="Both shares are combined with bitwise XOR. The receiver reconstructs the exact image only after "
         "AES-GCM authentication and a CNN share check succeed.",
)
recovery = "xor"

with st.sidebar.expander("Transmission faults (simulation)"):
    st.caption("Inject a problem on either channel to see how the receiver reacts.")
    faults = {}
    for index in range(1, n + 1):
        kind = st.selectbox(f"Channel {index}", FAULT_KINDS, key=f"fault_{index}_{n}",
                            format_func=lambda f: f"{f} – {FAULT_DESCRIPTIONS[f]}")
        if kind != "none":
            faults[index] = kind
    persistent = st.checkbox("Faults also hit retransmissions", value=False)
    retries = st.number_input("Retry budget (selective retransmission)", min_value=0, max_value=3, value=1)

with st.sidebar.expander("Advanced"):
    store_seed = st.checkbox("Store seed in log (enables bit-exact replay)", value=True,
                             help="Anyone with the seed and the input can regenerate the shares. "
                                  "Off = only a seed commitment is logged and replay is disabled.")
    show_bands = st.checkbox("Show seven spectral bands in session", value=True)
    secrecy = st.checkbox("Cross-session secrecy test (fresh random shares)", value=True)

run_clicked = st.sidebar.button("Run session", type="primary", disabled=image_source is None,
                                use_container_width=True)

st.title("Dynamic Visual Cryptography for Secure Image Transmission")
st.caption("A fixed *2-of-2* XOR split: Share A is fresh noise and Share B completes the image. "
           "Seven simulated spectral bands, AES-256-GCM packets, GCM authentication, and a CNN share check.")

tab_run, tab_history, tab_compare, tab_help = st.tabs(["Session", "History & replay", "Compare sessions", "How it works"])

if run_clicked and image_source is not None:
    config = FlowConfig(k=k, n=n, recovery=recovery, faults=faults, persistent_faults=persistent,
                        max_retries=int(retries), store_seed=store_seed, compute_secrecy=secrecy)
    with st.spinner("Running all seven stages ..."):
        st.session_state["result"] = run_flow_session(
            image_source, config, get_model(), WORKDIR, label=image_label
        )
        st.session_state["label"] = image_label

with tab_run:
    result = st.session_state.get("result")
    if result is None:
        st.info("Choose an image and settings in the sidebar, then press *Run session*.")
    else:
        config = result.config
        head = st.columns([2, 1, 1, 1])
        if result.ok:
            if result.gates.get("checks") == "passed":
                head[0].success(f"Session *{result.session_id}* delivered and verified.")
            else:
                head[0].warning(f"Session *{result.session_id}* delivered, but the receiver checks flagged a share. "
                                "See the checks gate below.")
        else:
            error = result.error
            head[0].error(f"Session aborted with *{error['code']}* in phase {error['phase']}.\n\n"
                          f"{error['message']}\n\n{EXPLANATIONS.get(error['code'], '')}")
        head[1].metric("Threshold", f"({config.k}, {config.n})")
        head[2].metric("Mode", config.recovery.upper())
        total = sum(v for key, v in result.timings.items() if key.endswith("_ms"))
        head[3].metric("Session time", f"{total:.0f} ms")

        if (result.secret is not None and result.bands is not None
                and result.share_a is not None and result.share_b is not None):
            st.html(diagram_html(result))

        st.subheader("Pipeline gates")
        gate_cols = st.columns(len(FLOW_PHASES))
        for column, (phase, title) in zip(gate_cols, FLOW_PHASES):
            state = result.gates.get(phase, "skipped")
            timing = result.timings.get(f"{phase}_ms")
            column.markdown(f"{GATE_ICON.get(state, '')} *{title}*  \n{state}"
                            + (f" · {timing:.1f} ms" if timing is not None else ""))

        if result.secret is not None:
            st.subheader("Secret image")
            info = st.columns([1, 2])
            info[0].image(to_png(result.secret.array), caption=f"{st.session_state.get('label', '')}", width="stretch")
            info[1].markdown(
                f"- Detected mode: *{result.secret.mode}* (source {result.secret.source_mode})\n"
                f"- Size: *{result.secret.width} × {result.secret.height}*, {result.secret.channels} channel(s)\n"
                f"- EXIF orientation applied: {fmt(result.secret.orientation_applied)}; "
                f"alpha flattened: {fmt(result.secret.alpha_flattened)}\n"
                f"- Input SHA-256: {result.source_digest[:32]}…\n"
                f"- Share scheme: *XOR (2-of-2, exact)*\n"
                f"- Spectral bands: *7 simulated* ({WAVELENGTHS_NM[0]}–{WAVELENGTHS_NM[-1]} nm)\n"
                + (f"- Session nonce: {result.key.nonce.hex()}\n- Seed commitment: {result.key.seed_commitment()[:32]}…"
                   if result.key else "")
            )
            if show_bands and result.bands is not None:
                st.markdown("*Spectral bands (simulated from RGB)*")
                band_columns = st.columns(len(WAVELENGTHS_NM))
                for column, index, wavelength in zip(band_columns, range(len(WAVELENGTHS_NM)), WAVELENGTHS_NM):
                    column.image(spectral_band_to_rgb(result.bands[..., index], wavelength),
                                 caption=f"{wavelength} nm", width="stretch")

        if result.shares:
            st.subheader("Shares (each one alone is noise)")
            tags = {entry["index"]: entry["tag"] for entry in result.manifest["tags"]} if result.manifest else {}
            columns = st.columns(len(result.shares))
            for column, share in zip(columns, result.shares):
                column.image(to_png(share.array), caption=f"Share {share.index}", width="stretch")
                column.caption(f"tag {tags.get(share.index, '')[:16]}… · "
                               f"{result.packet_bytes[share.index - 1]:,} ciphertext bytes")

        if result.receipts:
            st.subheader("Channels and admission")
            final = result.verification
            table = []
            for receipt in result.receipts:
                table.append({
                    "share": receipt["index"], "channel": receipt["channel"], "attempt": receipt["attempt"],
                    "transport": receipt["status"], "injected fault": receipt["fault"],
                    "bytes": receipt["bytes_written"],
                })
            left, right = st.columns(2)
            left.markdown("*Dispatch receipts*")
            left.dataframe(table, hide_index=True, width="stretch")
            if final:
                right.markdown(f"*Verification (attempt {len(result.attempts)})* — "
                               f"admitted {len(final.admitted)} of {final.n}, need {final.k}")
                right.dataframe([{"share": c.index, "status": f"{STATUS_ICON.get(c.status, '')} {c.status}",
                                  "detail": c.detail} for c in final.checks], hide_index=True, width="stretch")
                if len(result.attempts) > 1:
                    first = result.attempts[0]
                    right.caption("First attempt: " + ", ".join(f"#{c.index} {c.status}" for c in first.checks)
                                  + ". Only the failed shares were retransmitted.")
            st.caption("AES-256-GCM authenticates every share with a per-share HKDF-SHA256 key before decryption; "
                       "the master key is ephemeral and never leaves the sender/receiver boundary.")

        if result.ok:
            recon = result.reconstruction
            st.subheader("Reconstruction")
            columns = st.columns(3)
            columns[0].image(to_png(result.secret.array), caption="Original", width="stretch")
            columns[1].image(to_png(recon.display), caption=f"Recovered from shares {recon.used_indices}",
                             width="stretch")
            columns[2].image(to_png(result.below_threshold),
                             caption="Attacker's view: Share A alone", width="stretch")

            st.subheader("Scorecard")
            targets = result.scores["targets"]
            st.dataframe([{
                "metric": t["metric"], "value": fmt(t["value"]), "target": t["target"],
                "result": {True: "✅ pass", False: "❌ fail", None: "— n/a"}[t["passed"]], "note": t["note"],
            } for t in targets], hide_index=True, width="stretch")

            with st.expander("Per-share secrecy details"):
                st.dataframe([{
                    "share": item["index"], "entropy (bits)": round(item["entropy_bits"], 4),
                    "corr. to secret": round(item["corr_to_secret"], 4),
                    "adjacent corr. (h / v / d)": " / ".join(f"{v:.3f}" for v in item["adjacent_corr"].values()),
                } for item in result.scores["secrecy"]["per_share"]], hide_index=True, width="stretch")
                cross = result.scores["secrecy"].get("cross_session")
                if cross:
                    st.write(f"Cross-session test (same image, fresh random shares): NPCR min {cross['npcr_min']:.3f}%, "
                             f"UACI mean {cross['uaci_mean']:.3f}%, max |corr| {cross['max_abs_corr']:.4f}")

            with st.expander("Interactive stacking: choose any subset of shares"):
                chosen = st.multiselect("Shares to combine", [s.index for s in result.shares],
                                        default=[s.index for s in result.shares], key="subset")
                if chosen:
                    arrays = [result.shares[i - 1].array for i in chosen]
                    combined = arrays[0]
                    for array in arrays[1:]:
                        combined = np.bitwise_xor(combined, array)
                    enough = len(chosen) == 2
                    st.image(to_png(combined), width=420,
                             caption=("Both shares present: secret visible" if enough else "One share only: noise"))

            downloads = io.BytesIO()
            with zipfile.ZipFile(downloads, "w") as archive:
                for share in result.shares:
                    archive.writestr(f"share_{share.index}.png", to_png(share.array))
                if result.bands is not None:
                    for index, wavelength in enumerate(WAVELENGTHS_NM):
                        archive.writestr(f"band_{wavelength}nm.png",
                                         to_png(spectral_band_to_rgb(result.bands[..., index], wavelength)))
                archive.writestr("manifest.json", json.dumps(result.manifest, indent=2))
                archive.writestr("flow_report.json",
                                 json.dumps(report_payload(result, st.session_state.get("label", "")), indent=2, default=str))
                archive.writestr("reconstructed.png", to_png(recon.display))
            st.download_button("Download shares + manifest (.zip)", downloads.getvalue(),
                               file_name=f"dvc_{result.session_id}.zip", mime="application/zip")

with tab_history:
    records = read_flow_log(WORKDIR)
    if not records:
        st.info("No sessions logged yet.")
    else:
        st.caption(f"Log file: {WORKDIR / 'sessions.jsonl'} (one JSON line per session)")
        st.dataframe([{
            "session": r.get("session_id"), "status": r["status"],
            "error": (r.get("error") or {}).get("code", ""),
            "k": r["config"]["k"], "n": r["config"]["n"], "mode": r["config"]["recovery"],
            "image": r.get("image_mode", ""),
            "faults": ", ".join(f"#{i}:{f}" for i, f in r["config"].get("faults", {}).items()),
            "replayable": bool(r.get("seed")) and bool(r.get("tags")),
        } for r in reversed(records)], hide_index=True, width="stretch")

        replayable = [r["session_id"] for r in reversed(records) if r.get("seed") and r.get("tags")]
        if replayable:
            target = st.selectbox("Session to replay", replayable)
            if st.button("Replay bit-exact from log"):
                try:
                    outcome = replay_flow(target, get_model(), WORKDIR)
                    if outcome["bit_exact"]:
                        st.success(f"Replay of {target} is bit-exact: {outcome['tag_matches']} of "
                                   f"{outcome['tag_count']} share tags and the output digest match "
                                   f"({outcome['replay_ms']:.0f} ms).")
                    else:
                        st.error(f"Replay mismatch: tags {outcome['tag_matches']}/{outcome['tag_count']}, "
                                 f"output {outcome['output_match']}")
                    if outcome["reconstruction"] is not None:
                        st.image(to_png(outcome["reconstruction"].display), width=360, caption="Replayed output")
                except Exception as exc:
                    st.error(str(exc))

with tab_compare:
    st.markdown("Run the same image through a clean channel and a tampered packet, then compare the receiver's "
                "checks and recovery.")
    if image_source is None:
        st.info("Select an image in the sidebar first.")
    elif st.button("Run comparison: clean vs tampered"):
        rows = []
        previews = []
        progress = st.progress(0.0)
        settings = [("Clean channel", {}, 0), ("Tampered packet B", {2: "flip"}, 0)]
        total = len(settings)
        done = 0
        model = get_model()
        for label, injected, retry_budget in settings:
            res = run_flow_session(
                image_source,
                FlowConfig(k=2, n=2, recovery="xor", faults=injected, max_retries=retry_budget,
                           store_seed=False, compute_secrecy=False, log=False),
                model, WORKDIR, label=image_label,
            )
            done += 1
            progress.progress(done / total)
            gcm_passed = sum(1 for c in res.gcm_checks if c["verified"])
            cnn_passed = sum(1 for c in res.cnn_checks if c is not None and c["label"] == "genuine")
            rows.append({
                "setting": label, "session": res.session_id,
                "outcome": "recovered" if res.ok else f"blocked ({res.error['code']})",
                "PSNR dB": fmt(res.scores.get("fidelity", {}).get("psnr_db", 0.0)) if res.ok else "—",
                "GCM tags": f"{gcm_passed}/2", "CNN genuine": f"{cnn_passed}/2",
                "packet bytes": f"{sum(res.packet_bytes):,}",
                "check ms": round(res.timings.get("checks_ms", 0.0), 1),
            })
            previews.append((
                label,
                res.reconstruction.display if res.ok else res.below_threshold,
                res.ok,
            ))
        st.dataframe(rows, hide_index=True, width="stretch")
        columns = st.columns(len(previews) or 1)
        for column, (label, array, ok) in zip(columns, previews):
            column.image(to_png(array), caption=f"{label}: {'recovered' if ok else 'blocked — Share A only'}",
                         width="stretch")
        st.caption("The tampered packet fails GCM authentication and the receiver refuses to reconstruct.")

with tab_help:
    st.markdown("""
### What happens in a session

1. *Input image*: the image is read, rotated per EXIF, alpha is flattened onto white, and the mode (binary, gray,
   color) is detected. A SHA-256 digest binds the bytes to the session.
2. *Prism-inspired spectral preprocessing*: the RGB channels are mapped into seven simulated bands at
   400, 450, 500, 550, 600, 650 and 700 nm with fixed spectral weights. These bands are a teaching model, not
   measurements from a multispectral sensor.
3. *XOR share split*: Share A is fresh random noise and Share B is the source XOR Share A. The threshold is fixed
   at *(2, 2)*; both shares are required and one share alone reveals nothing.
4. *AES-256-GCM*: each share is sealed with a per-share key derived with HKDF-SHA256 from a fresh ephemeral master
   key, a 128-bit salt and a 96-bit nonce. GCM authenticates the ciphertext and metadata before decryption.
5. *Secure channel*: the two encrypted packets travel separately. A fault on a channel is detected and only the
   failed share is retransmitted within the retry budget.
6. *Receiver checks*: the receiver verifies both GCM tags and runs the ShareCNN classifier. GCM authentication is
   the hard gate; the CNN flags shares as *genuine*, *noisy* or *tampered* and is never used as the authenticator.
7. *XOR reconstruction*: the two authenticated shares are XORed back into the exact image.

### Choosing settings
* The threshold is fixed at *(2, 2)* because this flow needs both shares for exact XOR recovery.
* Use *Transmission faults* to flip, drop, delay, truncate, substitute or replay a packet. A drop or an
  authentication failure aborts the session unless a retry succeeds.
* *Store seed in log* enables bit-exact replay. Turn it off for sensitive images; only a seed commitment is logged.
* The cross-session test confirms that a fresh run produces unrelated shares (high NPCR/UACI, near-zero correlation).

### Scope and security note
The seven bands are simulated from RGB; this is a prism-inspired visualization, not measured spectroscopy.
AES-256-GCM provides confidentiality and integrity, while the CNN is a secondary demonstration check.
With "Store seed in log" on, runs/ contains everything needed to regenerate the shares. Treat it like the secret
image itself, or turn the option off for real data.
""")
