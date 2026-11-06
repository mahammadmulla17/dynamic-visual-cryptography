# Dynamic Visual Cryptography for Secure Image Transmission

The current dashboard implements a seven-stage, prism-inspired flow:

1. **Input image** - decode, normalize, flatten alpha, detect mode, and hash the source.
2. **Prism-inspired spectral preprocessing** - map RGB into seven simulated bands at 400, 450, 500, 550, 600,
   650 and 700 nm.
3. **XOR share split** - Share A is fresh noise; Share B is the source XOR Share A. The threshold is fixed at
   (2, 2), so both shares are required and either one alone is noise.
4. **AES-256-GCM** - seal each share with a per-share HKDF-SHA256 key, a 128-bit salt and a 96-bit nonce.
5. **Secure channel** - transmit the two encrypted packets separately, with optional fault injection and
   selective retransmission.
6. **Receiver checks** - verify both GCM tags and classify each share with a small CNN
   (`genuine`, `noisy`, `tampered`).
7. **XOR reconstruction** - combine the authenticated shares into the exact image.

The **Session** tab also renders a live seven-stage diagram, the spectral bands, the shares, the channel
receipts, the reconstruction, the attacker's view (Share A alone), and a scorecard. **History & replay** logs
sessions to `runs/sessions.jsonl` and can replay a stored session. **Compare sessions** runs a clean channel and
a tampered packet side by side. **How it works** explains the flow and its scope.

The 400-700 nm bands are **simulated from RGB**. This is a teaching visualization, not a measurement from a
multispectral sensor. AES-GCM provides confidentiality and integrity; the CNN is a secondary demo check.

## Requirements

- Python 3.12 or newer.
- Windows, macOS or Linux. The CNN runs on CPU.
- Dependencies are in `requirements.txt` (`numpy`, `Pillow`, `cryptography`, `torch`).

## How to run

Create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use
`.venv\Scripts\activate.bat` from Command Prompt.

Start the dashboard:

```bash
streamlit run app.py
```

Open http://localhost:8501. In the sidebar:

1. Choose a sample image or upload your own (PNG, JPEG, TIFF or BMP, up to 2048 x 2048).
2. The threshold is fixed at **(2, 2)** for this flow.
3. Optional: open **Transmission faults** and inject a problem on either channel
   (`drop`, `delay`, `flip`, `crop`, `substitute`, `replay`). Choose whether faults also hit retransmissions and
   set the retry budget.
4. Under **Advanced**, store the seed for replay, show the spectral bands, and enable the cross-session secrecy
   test.
5. Press **Run session**.

`main.py` is an alias entry point: `streamlit run main.py` launches the same app.

## Model

`share_cnn.pt` is a small convolutional classifier for 64 x 64 share crops. If the checkpoint is missing, the app
creates `data/shares.npz` (400 samples per class) and trains it on first use. To build or retrain it from the
command line:

```bash
python prism_flow.py
```

The trained model classifies clean shares as `genuine` with very high confidence on the included samples. The
scorecard reports both GCM and CNN results.

## Layout

```
app.py             current seven-stage dashboard
main.py            alias for app.py
prism_flow.py      spectral bands, XOR shares, AES-GCM, ShareCNN, training and CLI demo
flow_session.py    seven-stage session orchestration, faults, retries, logging and replay
diagram.py         SVG seven-stage diagram with live images and status badges
dvc/               original variable-threshold visual cryptography library
tests/             pytest suite for the original library
bench.py           benchmark for the original library
samples/           test images
share_cnn.pt       trained ShareCNN checkpoint
runs/              created at runtime (git-ignored): session log and replay inputs
```

## Security notes

- `runs/sessions.jsonl` stores the session seed by default so a session can be replayed. Seed plus input
  regenerates the shares. Treat `runs/` like the secret image, or turn off **Store seed in log**.
- The master key is ephemeral and is never logged. The log stores only a seed commitment when seed storage is
  disabled.
- AES-256-GCM authenticates the ciphertext and metadata before decryption. The CNN is not an authenticator.
- Channels are simulated locally. A live network transport, video, and multiple secrets per share set are out of
  scope.
- The seven spectral bands are simulated from RGB and are not a substitute for a real multispectral sensor.

## Original threshold dashboard

The `dvc/` package, tests and benchmark implement the earlier variable-threshold dashboard
(`k` of `n` shares, XOR/GF(256) or OR stacking, SHA-256 tags and manifests). They are kept for reference and for
the original test suite:

```bash
python -m pytest -q
python bench.py
```

The current `app.py` uses the new prism-inspired flow and does not call the old dashboard code.
