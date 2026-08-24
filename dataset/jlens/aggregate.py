"""Client-side aggregation of a `/api/lens/prompt` response into a top-k concept list.

The Neuronpedia API has no layer-range or frequency-sort option: a JACOBIAN_LENS
response always carries every fitted layer's top-N read-out tokens for every
position. Restricting to a later-layer window and ranking by frequency is
entirely a post-processing step on our side. This module is that step, kept as
a pure function of (raw response, config) so the derived top-k is always
reproducible from a saved raw response without re-calling the API.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_LENS_TYPE = "JACOBIAN_LENS"
AGGREGATION_VERSION = "frequency_over_selected_layers_and_answer_tokens_v1"


def normalize_token(token: str) -> str:
    return token.strip().lower()


@dataclass(frozen=True)
class JLensConfig:
    layer_min: int
    layer_max: int
    top_k: int = 10
    lens_type: str = DEFAULT_LENS_TYPE
    aggregation: str = AGGREGATION_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "lens_type": self.lens_type,
            "layer_min": self.layer_min,
            "layer_max": self.layer_max,
            "top_k": self.top_k,
            "aggregation": self.aggregation,
        }


def aggregate_top_k(response: dict, config: JLensConfig) -> list[dict[str, object]]:
    """Aggregate a buffered (`stream: false`) `/api/lens/prompt` response.

    Keeps only generated-token positions (`is_generated: true`) and read-out
    layers inside `[config.layer_min, config.layer_max]`, counts normalized
    token frequency across those slices, and returns the top-k by count.

    Ties break by earliest generated position the token was seen at, then
    lexically -- both arbitrary otherwise, but a fixed rule keeps the result
    reproducible.
    """
    layers_by_type = response["meta"]["layers_by_type"][config.lens_type]

    counts: dict[str, int] = {}
    first_position: dict[str, int] = {}
    layer_breakdown: dict[str, dict[int, int]] = {}

    for frame in response["tokens"]:
        if not frame["is_generated"]:
            continue
        for result in frame["results"]:
            if result["type"] != config.lens_type:
                continue
            for layer, layer_tokens in zip(layers_by_type, result["top_tokens"]):
                if not (config.layer_min <= layer <= config.layer_max):
                    continue
                for raw_token in layer_tokens:
                    token = normalize_token(raw_token)
                    counts[token] = counts.get(token, 0) + 1
                    layer_breakdown.setdefault(token, {})
                    layer_breakdown[token][layer] = layer_breakdown[token].get(layer, 0) + 1
                    first_position.setdefault(token, frame["position"])

    ordered = sorted(
        counts.items(),
        key=lambda item: (-item[1], first_position[item[0]], item[0]),
    )

    return [
        {
            "concept": token,
            "count": count,
            "rank": rank,
            "first_generated_position": first_position[token],
            "layers": dict(sorted(layer_breakdown[token].items())),
        }
        for rank, (token, count) in enumerate(ordered[: config.top_k], start=1)
    ]
