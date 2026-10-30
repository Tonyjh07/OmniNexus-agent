"""Model registry and capability negotiation (routing only).

v0.2：per-provider HTTP adapters were replaced by the Pydantic-AI based
``agent`` engine. This module keeps the pure config-level routing logic:
requested model → ordered candidate chain, capability routing/degradation and
global reroute. Execution (fallback, retry, streaming) lives in
:mod:`omninexus.agent.engine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omninexus.config import AppConfig, ModelConfig
from omninexus.core.models import ContentPart
from omninexus.errors import ErrorCode, OmniError


@dataclass(slots=True)
class RoutePlan:
    requested_model: str
    primary: ModelConfig
    candidates: list[ModelConfig]
    effective_parts: list[ContentPart]
    degraded: bool = False
    capability_reason: str | None = None
    notes: list[str] = field(default_factory=list)
    selected_model: str | None = None
    selected_provider: str | None = None
    fallback_used: bool = False
    attempts: list[dict] = field(default_factory=list)

    @property
    def selected_model_config(self) -> ModelConfig:
        if not self.selected_model:
            return self.primary
        return next(
            (model for model in self.candidates if model.id == self.selected_model),
            self.primary,
        )


class GenerationInterruptedError(OmniError):
    """保留类型：旧 gateway 的流中断语义（新引擎用 DoneFrame.interrupted 表达）。"""

    def __init__(self, message: str) -> None:
        super().__init__(
            ErrorCode.GENERATION_INTERRUPTED,
            message,
            status_code=502,
            retryable=False,
        )


class Gateway:
    """Model registry + capability routing. 只做「选哪个模型」，不做网络调用。"""

    def __init__(self, config: AppConfig, *, transport=None) -> None:
        self.config = config
        self._transport = transport  # 保留签名兼容；v0.2 不再用于模型层

    async def close(self) -> None:  # 无持有性资源
        return None

    def _chain(self, model_id: str) -> list[ModelConfig]:
        chain: list[ModelConfig] = []
        seen: set[str] = set()

        def walk(current_id: str, depth: int = 0) -> None:
            if depth > 8 or current_id in seen:
                return
            model = self.config.model_config_by_id(current_id)
            if not model:
                return
            seen.add(current_id)
            chain.append(model)
            for fallback in model.fallbacks:
                walk(fallback, depth + 1)

        walk(model_id)
        for fallback in self.config.gateway.fallback_models:
            walk(fallback)
        return chain

    @staticmethod
    def _supports(model: ModelConfig, parts: list[ContentPart]) -> bool:
        required = {part.modality for part in parts if part.modality != "text"}
        return required.issubset(set(model.input_modalities))

    @staticmethod
    def _degrade_part(part: ContentPart) -> ContentPart:
        metadata = part.metadata
        if part.modality == "image":
            alt = metadata.get("alt") or metadata.get("description") or metadata.get("ocr_text")
            name = metadata.get("filename") or metadata.get("name")
            if alt:
                text = f"[Image content (text description): {str(alt)[:2000]}]"
            elif name:
                text = f"[Image attachment omitted: {str(name)[:200]}]"
            else:
                text = "[An image was provided, but the selected model does not accept images.]"
        elif part.modality == "audio":
            transcript = metadata.get("transcript") or metadata.get("text")
            text = (
                f"[Audio transcript: {str(transcript)[:4000]}]"
                if transcript
                else (
                    "[An audio attachment was provided, but the selected model "
                    "does not accept audio.]"
                )
            )
        elif part.modality == "file":
            extracted = metadata.get("extracted_text") or metadata.get("text")
            name = metadata.get("filename") or metadata.get("name") or "file"
            text = (
                f"[File {str(name)[:200]} contents: {str(extracted)[:4000]}]"
                if extracted
                else f"[File attachment omitted: {str(name)[:200]}]"
            )
        else:
            text = f"[{part.modality} attachment omitted: unsupported modality.]"
        return ContentPart(
            modality="text",
            content=text,
            metadata={
                key: value
                for key, value in metadata.items()
                if key in ("alt", "description", "filename", "name", "mime_type")
            },
        )

    def plan(self, model_id: str, parts: list[ContentPart]) -> RoutePlan:
        model = self.config.model_config_by_id(model_id)
        if not model:
            raise OmniError(
                ErrorCode.UNKNOWN_MODEL,
                "Unknown model id.",
                status_code=404,
                details={"model": model_id},
            )
        chain = self._chain(model_id)
        if not chain:
            raise OmniError(ErrorCode.UNKNOWN_MODEL, "Unknown model id.", status_code=404)

        all_configured = [
            model
            for model in self.config.models
            if self.config.providers[model.provider].is_configured()
        ]
        configured = [candidate for candidate in chain if candidate in all_configured]
        global_reroute = False
        if not configured:
            if all_configured:
                chain = all_configured
                configured = all_configured
                global_reroute = True
            else:
                raise OmniError(
                    ErrorCode.NO_PROVIDER_AVAILABLE,
                    "No configured model is available. Add a provider key to .env "
                    "(or enable the offline echo model) and restart.",
                    status_code=503,
                    details={
                        "requested_model": model_id,
                        "models": [
                            {
                                "id": item.id,
                                "available": self.config.model_info(item).available,
                                "reason": self.config.model_info(item).unavailable_reason,
                            }
                            for item in chain[:8]
                        ],
                    },
                )

        base = chain[0] if chain[0] in configured else configured[0]
        effective_parts = list(parts)
        degraded = False
        reason: str | None = None
        notes: list[str] = []

        if global_reroute:
            reason = "requested_unavailable"
            notes.append(
                f"requested model chain for {model_id} has no configured provider; "
                f"using configured model {base.id}"
            )
        elif base is not chain[0]:
            reason = "requested_unavailable"
            notes.append(f"requested model {chain[0].id} is unavailable; using {base.id}")

        if not self._supports(base, parts):
            required = {p.modality for p in parts if p.modality != "text"}
            vision_capable = next(
                (item for item in all_configured if self._supports(item, parts)), None
            )
            if vision_capable is not None:
                base = vision_capable
                if vision_capable not in configured:
                    configured.append(vision_capable)
                for candidate in all_configured:
                    if candidate not in configured:
                        configured.append(candidate)
                reason = "capability_route"
                notes.append(f"routed to {base.id} for input modalities {sorted(required)}")
            else:
                effective_parts = [self._degrade_part(part) for part in parts]
                degraded = True
                reason = "capability_degraded"
                notes.append(f"model {base.id} lacks {sorted(required)}; degraded to text")

        if degraded and not any(p.modality == "text" for p in effective_parts):
            effective_parts.append(ContentPart(modality="text", content="[Unsupported attachment]"))

        ordered: list[ModelConfig] = []
        seen: set[str] = set()
        for candidate in self._chain(base.id) + configured:
            if candidate.id not in seen and candidate in configured:
                seen.add(candidate.id)
                ordered.append(candidate)

        if not ordered:
            raise OmniError(
                ErrorCode.NO_PROVIDER_AVAILABLE,
                "No configured model is available.",
                status_code=503,
            )
        return RoutePlan(
            requested_model=model_id,
            primary=base,
            candidates=ordered,
            effective_parts=effective_parts,
            degraded=degraded,
            capability_reason=reason,
            notes=notes,
        )
