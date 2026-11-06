"""Shared definitions for the long-paper figures."""
from __future__ import annotations
from . import natstyle as NS

# Okabe-Ito based vendor palette; Microsoft uses a dark gold because Okabe-Ito
# yellow is illegible on white.
VENDOR = {"OpenAI": "#0072B2", "Anthropic": "#E69F00", "xAI": "#CC79A7",
          "Meta": "#009E73", "Moonshot": "#D55E00", "Mistral": "#56B4E9",
          "Microsoft": "#8C6D1F", "Google": "#6A3D9A"}
HUMAN, HUMAN_FILL, FEIGNER, POLE = "#2B2B2B", "#C8C8C8", "#6E6E6E", "#B22222"

# canonical order (vendor, then size) used on every categorical axis
MODELS = [  # label, short, vendor, pcl5/logprob tag
    ("GPT-4o-mini", "GPT-4o-mini", "OpenAI", "gpt4omini"),
    ("GPT-4o", "GPT-4o", "OpenAI", "gpt4o"),
    ("GPT-4.1-nano", "GPT-4.1-nano", "OpenAI", "gpt41nano"),
    ("GPT-4.1-mini", "GPT-4.1-mini", "OpenAI", "gpt41mini"),
    ("GPT-4.1", "GPT-4.1", "OpenAI", "gpt41"),
    ("GPT-5-mini", "GPT-5-mini", "OpenAI", "gpt5mini"),
    ("GPT-5.2", "GPT-5.2", "OpenAI", "gpt52"),
    ("GPT-5.4-mini", "GPT-5.4-mini", "OpenAI", "gpt54mini"),
    ("Claude Haiku 4.5", "Haiku 4.5", "Anthropic", "claudehaiku45"),
    ("Grok-4.1-fast", "Grok-4.1", "xAI", "grok41fastnonreasoning"),
    ("Llama-3.3-70B", "Llama-3.3", "Meta", "llama3370binstruct"),
    ("Kimi-K2.5", "Kimi-K2.5", "Moonshot", "kimik25"),
    ("Mistral-Large-3", "Mistral-L3", "Mistral", "mistrallarge3"),
    ("Phi-4", "Phi-4", "Microsoft", "phi4"),
    ("Gemini-2.5-Flash-Lite", "Gemini-2.5-FL", "Google", "gemini25flashlite"),
    ("Gemini-2.5-Flash", "Gemini-2.5-F", "Google", "gemini25flash"),
    ("Gemini-2.5-Pro", "Gemini-2.5-Pro", "Google", "gemini25pro"),
]
SHORT = {m[0]: m[1] for m in MODELS}
VENDOR_OF = {m[0]: m[2] for m in MODELS}
TAG2LABEL = {m[3]: m[0] for m in MODELS}
ORDER = [m[0] for m in MODELS]

def color(label: str) -> str:
    return VENDOR[VENDOR_OF[label]]

def vendor_handles(vendors=None, ms=3.4):
    from matplotlib.lines import Line2D
    vs = vendors or list(VENDOR)
    return [Line2D([], [], marker="o", ls="", ms=ms, mfc=VENDOR[v], mec="w", mew=.4, label=v)
            for v in vs]

OUT = None
def save(fig, name):
    from .config import FIGURES
    d = FIGURES / "long"; d.mkdir(parents=True, exist_ok=True)
    NS.save(fig, d / f"{name}.png")
