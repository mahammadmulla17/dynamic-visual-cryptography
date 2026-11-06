"""Render the seven-stage Dynamic Visual Cryptography workflow as an SVG diagram."""

from __future__ import annotations

import base64
import html
from io import BytesIO

import numpy as np
from PIL import Image

from prism_flow import FlowResult, WAVELENGTHS_NM

NAVY = "#17365d"
INK = "#5c6b82"
BLUE = "#3a6db5"
ARROW = "#8496b3"
CARD = "#ffffff"
BG = "#eef3fb"
FONT = "'Segoe UI', 'Inter', Arial, sans-serif"
SERIF = "Georgia, 'Times New Roman', serif"


def _image_uri(array: np.ndarray, size: int, resample: Image.Resampling) -> str:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), resample)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _text(x: float, y: float, value: str, **attrs) -> str:
    options = {
        "x": x,
        "y": y,
        "text-anchor": "middle",
        "font-family": FONT,
        "font-size": 13.5,
        "fill": INK,
    }
    options.update({key.replace("_", "-"): value for key, value in attrs.items()})
    body = html.escape(value)
    attributes = " ".join(
        f'{key}="{html.escape(str(value))}"' for key, value in options.items()
    )
    return f"<text {attributes}>{body}</text>"


def _card(x: float, y: float, width: float, height: float) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" '
        f'fill="{CARD}" filter="url(#cardShadow)"/>'
    )


def _arrow(x1: float, x2: float, y: float) -> str:
    return (
        f'<line x1="{x1}" y1="{y}" x2="{x2 - 8}" y2="{y}" stroke="{ARROW}" '
        f'stroke-width="5" stroke-linecap="round"/>'
        f'<polygon points="{x2},{y - 9} {x2 - 13},{y} {x2},{y + 9}" fill="{ARROW}"/>'
    )


def _lock(cx: float) -> str:
    return (
        f'<rect x="{cx - 25}" y="291" width="50" height="45" rx="7" fill="{BLUE}"/>'
        f'<path d="M {cx - 15} 291 v -13 a 15 15 0 0 1 30 0 v 13" fill="none" '
        f'stroke="{BLUE}" stroke-width="9" stroke-linecap="round"/>'
        f'<circle cx="{cx}" cy="310" r="6" fill="{BG}"/>'
        f'<rect x="{cx - 3.5}" y="312" width="7" height="13" rx="3" fill="{BG}"/>'
    )


def _network(cx: float) -> str:
    return (
        f'<line x1="{cx - 44}" y1="301" x2="{cx - 16}" y2="301" stroke="{BLUE}" '
        f'stroke-width="4" stroke-dasharray="7 6" stroke-linecap="round"/>'
        f'<line x1="{cx + 8}" y1="301" x2="{cx + 34}" y2="301" stroke="{BLUE}" '
        f'stroke-width="4" stroke-dasharray="7 6" stroke-linecap="round"/>'
        f'<circle cx="{cx - 53}" cy="301" r="8" fill="{BLUE}"/>'
        f'<circle cx="{cx - 4}" cy="301" r="8" fill="{BLUE}"/>'
        f'<polygon points="{cx + 55},301 {cx + 35},290 {cx + 35},312" fill="{BLUE}"/>'
    )


def _check(cx: float, cy: float, color: str = "#2e9b57") -> str:
    return (
        f'<path d="M {cx - 24} {cy} l 17 17 l 33 -37" fill="none" stroke="{color}" '
        f'stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def _cross(cx: float, cy: float, color: str = "#c2413b") -> str:
    return (
        f'<path d="M {cx - 18} {cy - 18} l 36 36 M {cx + 18} {cy - 18} l -36 36" '
        f'fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round"/>'
    )


def _cnn_grid(cx: float, cy: float, color: str = BLUE) -> str:
    squares = []
    for row in range(2):
        for col in range(3):
            x = cx - 38 + col * 27
            y = cy - 20 + row * 27
            squares.append(f'<rect x="{x}" y="{y}" width="21" height="21" rx="5" fill="{color}"/>')
    return "".join(squares)


def _xor(cx: float, cy: float) -> str:
    return (
        f'<circle cx="{cx}" cy="{cy}" r="15" fill="none" stroke="{BLUE}" stroke-width="2.5"/>'
        f'<line x1="{cx - 8}" y1="{cy}" x2="{cx + 8}" y2="{cy}" stroke="{BLUE}" stroke-width="2.5"/>'
        f'<line x1="{cx}" y1="{cy - 8}" x2="{cx}" y2="{cy + 8}" stroke="{BLUE}" stroke-width="2.5"/>'
    )


def build_flow_svg(result: FlowResult) -> str:
    gcm_ok = all(check["verified"] for check in result.gcm_checks)
    cnn_ok = all(check is not None and check["label"] == "genuine" for check in result.cnn_checks)
    recovered = result.reconstructed is not None

    source_uri = _image_uri(result.source, 106, Image.Resampling.LANCZOS)
    share_a_uri = _image_uri(result.share_a, 62, Image.Resampling.NEAREST)
    share_b_uri = _image_uri(result.share_b, 62, Image.Resampling.NEAREST)
    share_a_large = _image_uri(result.share_a, 70, Image.Resampling.NEAREST)
    recovered_uri = (
        _image_uri(result.reconstructed, 70, Image.Resampling.LANCZOS) if recovered else None
    )

    spectrum_stops = "".join(
        f'<stop offset="{index / (len(WAVELENGTHS_NM) - 1) * 100:.2f}%" '
        f'stop-color="{color}"/>'
        for index, color in enumerate(
            ("#5b2ea6", "#3b4cc0", "#2f80c8", "#2eb872", "#c9d12e", "#f2b21b", "#e87020")
        )
    )

    labels = []
    for index, wavelength in enumerate(WAVELENGTHS_NM):
        x = 355 + index * 35
        labels.append(_text(x, 309, str(wavelength), font_size=10.5))
    labels.append(_text(593, 309, "nm", font_size=10.5))

    gcm_fill = "#e8f5e9" if gcm_ok else "#fdecea"
    gcm_mark = _check(923, 594) if gcm_ok else _cross(923, 594)
    cnn_fill = "#eef2fb" if cnn_ok else "#fff3e2"
    cnn_mark = _cnn_grid(1068, 594) if cnn_ok else _cross(1068, 594, "#c9832c")

    if recovered_uri:
        recovery_panel = (
            f'<image x="735" y="552" width="70" height="70" href="{recovered_uri}" '
            f'clip-path="url(#clip7b)"/>'
        )
        recovery_label = _text(770, 649, "Recovered image", font_size=12.5)
    else:
        recovery_panel = (
            '<rect x="735" y="552" width="70" height="70" rx="8" fill="#f3f5f9" '
            'stroke="#d8dee9" stroke-width="1.5"/>'
            '<text x="770" y="590" text-anchor="middle" font-family="' + FONT + '" '
            'font-size="11" fill="#8b98ab">blocked</text>'
        )
        recovery_label = _text(770, 649, "Reconstruction blocked", font_size=11.5)

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="900" viewBox="0 0 1440 900" role="img"
 aria-label="Seven-stage Dynamic Visual Cryptography workflow" style="width:100%;height:auto;display:block">
<defs>
  <filter id="cardShadow" x="-20%" y="-20%" width="140%" height="160%">
    <feDropShadow dx="0" dy="7" stdDeviation="9" flood-color="#1b3a66" flood-opacity="0.10"/>
  </filter>
  <linearGradient id="spectrum" x1="0" y1="0" x2="1" y2="0">{spectrum_stops}</linearGradient>
  <clipPath id="clip1"><rect x="114.5" y="241" width="106" height="106" rx="14"/></clipPath>
  <clipPath id="clip3a"><rect x="651" y="253" width="62" height="62" rx="7"/></clipPath>
  <clipPath id="clip3b"><rect x="731" y="253" width="62" height="62" rx="7"/></clipPath>
  <clipPath id="clip7a"><rect x="470" y="552" width="70" height="70" rx="8"/></clipPath>
  <clipPath id="clip7b"><rect x="735" y="552" width="70" height="70" rx="8"/></clipPath>
</defs>

<rect width="1440" height="900" fill="{BG}"/>
{_text(720, 64, "Dynamic Visual Cryptography", font_family=SERIF, font_size=40, font_weight=700, fill=NAVY)}
{_text(720, 98, "Secure image transmission · MCA project workflow", font_size=17, fill="#6b7a90")}

{_card(57, 189, 221, 216)}
{_text(167.5, 224, "1 · Input image", font_size=17, font_weight=700, fill=NAVY)}
<rect x="114.5" y="241" width="106" height="106" rx="14" fill="#1f2a44"/>
<image x="114.5" y="241" width="106" height="106" href="{source_uri}" clip-path="url(#clip1)"/>
{_text(167.5, 375, "RGB source")}

{_card(327, 189, 234, 216)}
{_text(444, 222, "2 · Prism-inspired", font_size=17, font_weight=700, fill=NAVY)}
{_text(444, 244, "spectral preprocessing", font_size=13.5, fill="#344a63")}
<rect x="355" y="271" width="178" height="21" rx="10.5" fill="url(#spectrum)"/>
{''.join(labels)}
{_text(444, 340, "7 simulated bands from RGB")}

{_card(608, 189, 229, 216)}
{_text(722.5, 222, "3 · XOR share split", font_size=17, font_weight=700, fill=NAVY)}
<image x="651" y="253" width="62" height="62" href="{share_a_uri}" clip-path="url(#clip3a)"/>
<image x="731" y="253" width="62" height="62" href="{share_b_uri}" clip-path="url(#clip3b)"/>
{_text(682, 337, "Share A")}
{_text(762, 337, "Share B")}
{_text(722.5, 367, "Both required to reconstruct")}

{_card(884, 189, 231, 216)}
{_text(999.5, 222, "4 · AES-256-GCM", font_size=17, font_weight=700, fill=NAVY)}
{_lock(999.5)}
{_text(999.5, 350, "Encrypt + authenticate", font_size=12.8)}
{_text(999.5, 367, "before transmission", font_size=12.8)}

{_card(1162, 189, 221, 216)}
{_text(1272.5, 222, "5 · Secure channel", font_size=17, font_weight=700, fill=NAVY)}
{_network(1272.5)}
{_text(1272.5, 350, "Ciphertext packets", font_size=12.8)}
{_text(1272.5, 367, "over an untrusted link", font_size=12.8)}

{_arrow(292, 327, 297)}
{_arrow(575, 608, 297)}
{_arrow(851, 884, 297)}
{_arrow(1129, 1162, 297)}

{_card(410, 508, 340, 186)}
{_text(580, 537, "7 · XOR reconstruction", font_size=17, font_weight=700, fill=NAVY)}
<image x="470" y="552" width="70" height="70" href="{share_a_large}" clip-path="url(#clip7a)"/>
{_xor(610, 587)}
{recovery_panel}
{_text(505, 649, "Share A", font_size=12.5)}
{recovery_label}

{_card(825, 508, 340, 186)}
{_text(995, 537, "6 · Receiver checks", font_size=17, font_weight=700, fill=NAVY)}
<rect x="858" y="552" width="130" height="84" rx="10" fill="{gcm_fill}"/>
{gcm_mark}
<rect x="1003" y="552" width="130" height="84" rx="10" fill="{cnn_fill}"/>
{cnn_mark}
{_text(923, 661, "GCM tag verified" if gcm_ok else "GCM tag failed", font_size=12.5)}
{_text(1068, 661, "CNN share check" if cnn_ok else "CNN flagged share", font_size=12.5)}

<path d="M 1272.5 405 V 447 H 1002" fill="none" stroke="{ARROW}" stroke-width="4" stroke-linecap="round"/>
<polygon points="990,447 1003,438 1003,456" fill="{ARROW}"/>
<line x1="825" y1="590" x2="762" y2="590" stroke="{ARROW}" stroke-width="4" stroke-linecap="round"/>
<polygon points="750,590 763,581 763,599" fill="{ARROW}"/>

<rect x="57" y="786" width="1326" height="78" rx="14" fill="#fff6e4" stroke="#f0d9a8" stroke-width="1"/>
{_text(82, 812, "Important project scope", text_anchor="start", font_size=15, font_weight=700, fill="#7a4d12")}
{_text(82, 838, "Seven bands are simulated from RGB, not measured by a multispectral sensor. AES-GCM provides security; CNN predictions are a secondary demo check and never replace authentication.", text_anchor="start", font_size=13.5, fill="#8c6a3a")}
</svg>'''
