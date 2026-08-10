"""Workload model: profiles, length distributions, and request generation.

A *profile* (a.k.a. workload class) fixes how prompt length and target output length are drawn,
plus a ``spec_decode`` flag. Profiles model different tenants:

  short      — chat-style: short prompt, short output.
  long       — summarization/RAG: long prompt, moderate output.
  reasoning  — moderate prompt, long output (reasoning-like).
  coding     — code continuation: repetitive/structured prompt whose output reuses prompt n-grams,
               so n-gram speculative decoding gets high acceptance. This is our *spec-decode class*.

Note on ``spec_decode``: vLLM 0.8.5 configures speculative decoding at *server startup*, not per
request, so this flag does NOT toggle the engine per request. It is (a) intent metadata for
per-class metric breakdown (acceptance differs sharply by class), and (b) the signal a Phase-2
admission policy will schedule on. The *coding* profile is authored to actually exercise n-gram
lookup so the measured acceptance_rate reflects the flag.

All draws take an explicit ``random.Random`` so runs are reproducible under a fixed seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# --- small text corpora for building prompts of a target length -------------------------

# A prose paragraph (repeated/truncated to hit a target word count) for text-like classes.
_PROSE = (
    "Large language model inference is dominated by memory bandwidth during the decode phase. "
    "Each generated token requires reading the full set of model weights and the growing key-value "
    "cache, so throughput is bounded by how fast the device can move bytes rather than by raw "
    "arithmetic. Speculative decoding proposes several candidate tokens per step and verifies them "
    "in parallel, trading a little extra compute for fewer sequential steps. Admission and ordering "
    "policies placed in front of the serving engine shape tail latency when many tenants compete. "
).split()

# A repetitive code-like block: its output continuations reuse prompt n-grams, so n-gram
# prompt-lookup speculation lands often -> higher acceptance for the coding/spec-decode class.
_CODE = (
    "def process(items):\n"
    "    results = []\n"
    "    for item in items:\n"
    "        value = transform(item)\n"
    "        if value is not None:\n"
    "            results.append(value)\n"
    "    return results\n"
).split()


@dataclass
class LenDist:
    """A length distribution in tokens. ``kind`` is 'fixed' | 'uniform' | 'lognormal'."""

    kind: str
    lo: int = 1
    hi: int = 1
    mu: float = 0.0  # lognormal: mean of underlying normal (in log tokens)
    sigma: float = 0.0

    def sample(self, rng: random.Random) -> int:
        if self.kind == "fixed":
            return int(self.lo)
        if self.kind == "uniform":
            return rng.randint(int(self.lo), int(self.hi))
        if self.kind == "lognormal":
            val = int(round(rng.lognormvariate(self.mu, self.sigma)))
            return max(int(self.lo), min(int(self.hi), val))
        raise ValueError(f"unknown length dist kind: {self.kind!r}")

    @classmethod
    def from_cfg(cls, d: dict) -> "LenDist":
        return cls(
            kind=d.get("kind", "fixed"),
            lo=d.get("lo", d.get("min", 1)),
            hi=d.get("hi", d.get("max", 1)),
            mu=d.get("mu", 0.0),
            sigma=d.get("sigma", 0.0),
        )


@dataclass
class Profile:
    name: str
    prompt_len: LenDist
    output_len: LenDist
    spec_decode: bool = False
    corpus: str = "prose"  # 'prose' | 'code'

    @classmethod
    def from_cfg(cls, name: str, d: dict) -> "Profile":
        return cls(
            name=name,
            prompt_len=LenDist.from_cfg(d["prompt_len"]),
            output_len=LenDist.from_cfg(d["output_len"]),
            spec_decode=bool(d.get("spec_decode", False)),
            corpus=d.get("corpus", "prose"),
        )


@dataclass
class RequestSpec:
    """One concrete request to send."""

    profile: str
    spec_decode: bool
    prompt: str
    max_tokens: int
    prompt_words: int  # approx requested prompt length (actual token count comes from usage)
    seq: int = field(default=0)


def _build_prompt(words: int, corpus: str) -> str:
    """Assemble a prompt of ~``words`` words by cycling the chosen corpus."""
    bank = _CODE if corpus == "code" else _PROSE
    if words <= 0:
        words = 1
    reps = (words // len(bank)) + 1
    toks = (bank * reps)[:words]
    body = " ".join(toks)
    if corpus == "code":
        return "Continue this Python module in the same style:\n\n" + body + "\n"
    return "Summarize and continue the following passage:\n\n" + body + "\n"


def sample_request(profile: Profile, rng: random.Random, seq: int = 0) -> RequestSpec:
    """Draw one request from a profile using ``rng`` (reproducible under a fixed seed)."""
    pwords = profile.prompt_len.sample(rng)
    otoks = profile.output_len.sample(rng)
    return RequestSpec(
        profile=profile.name,
        spec_decode=profile.spec_decode,
        prompt=_build_prompt(pwords, profile.corpus),
        max_tokens=otoks,
        prompt_words=pwords,
        seq=seq,
    )


# --- default profile set (overridable per config) ---------------------------------------

DEFAULT_PROFILES: dict[str, Profile] = {
    "short": Profile("short", LenDist("uniform", 16, 64), LenDist("uniform", 16, 64), False, "prose"),
    "long": Profile("long", LenDist("uniform", 400, 900), LenDist("uniform", 64, 160), False, "prose"),
    "reasoning": Profile("reasoning", LenDist("uniform", 80, 200), LenDist("uniform", 200, 400), False, "prose"),
    "coding": Profile("coding", LenDist("uniform", 120, 300), LenDist("uniform", 100, 256), True, "code"),
}


def load_profiles(cfg) -> dict[str, Profile]:
    """Build the profile table from a config's ``workload.profiles`` map, falling back to defaults."""
    wl = cfg.get("workload", {}) or {}
    prof_cfg = wl.get("profiles")
    if not prof_cfg:
        return dict(DEFAULT_PROFILES)
    return {name: Profile.from_cfg(name, d) for name, d in prof_cfg.items()}


class WorkloadMix:
    """Weighted set of profiles; ``pick`` chooses a profile per arrival (reproducibly)."""

    def __init__(self, profiles: dict[str, Profile], weights: dict[str, float]):
        self.profiles = profiles
        names = list(weights.keys())
        for n in names:
            if n not in profiles:
                raise KeyError(f"mix references unknown profile {n!r}")
        self._names = names
        self._weights = [weights[n] for n in names]

    def pick(self, rng: random.Random) -> Profile:
        name = rng.choices(self._names, weights=self._weights, k=1)[0]
        return self.profiles[name]

    @classmethod
    def from_cfg(cls, cfg) -> "WorkloadMix":
        wl = cfg.get("workload", {}) or {}
        mix = wl.get("mix") or {"short": 1.0}
        return cls(load_profiles(cfg), mix)
