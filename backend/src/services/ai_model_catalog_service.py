"""
Provider-backed model catalog lookup for AI settings.
"""
from __future__ import annotations

import json
from typing import Any, Iterable
from urllib import error, parse, request

SUPPORTED_AI_PROVIDERS = {"openai", "google", "anthropic", "zai"}
ZAI_OPENAI_BASE_URL = "https://api.z.ai/api/paas/v4"
HTTP_TIMEOUT_SECONDS = 15


class ModelCatalogError(Exception):
    """Raised when a provider model catalog cannot be retrieved."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _dedupe_and_sort(values: Iterable[str]) -> list[str]:
    unique = {value.strip() for value in values if isinstance(value, str) and value.strip()}
    return sorted(unique)


def _extract_error_message(payload: Any) -> str | None:
    if isinstance(payload, dict):
        # Common provider error shapes:
        # {"error": {"message": "..."}}, {"message": "..."} or {"detail": "..."}
        nested_error = payload.get("error")
        if isinstance(nested_error, dict):
            nested_message = nested_error.get("message")
            if isinstance(nested_message, str) and nested_message.strip():
                return nested_message.strip()
        for key in ("message", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _request_json(provider: str, url: str, headers: dict[str, str]) -> dict[str, Any]:
    req = request.Request(url, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        message = ""
        if body:
            try:
                parsed = json.loads(body)
                message = _extract_error_message(parsed) or body
            except json.JSONDecodeError:
                message = body
        message = (message or f"HTTP {exc.code}").strip()
        if exc.code in {401, 403}:
            raise ModelCatalogError(
                f"{provider} API key is invalid or unauthorized ({message})",
                status_code=401,
            ) from exc
        if exc.code == 429:
            raise ModelCatalogError(
                f"{provider} model listing was rate-limited ({message})",
                status_code=429,
            ) from exc
        raise ModelCatalogError(
            f"{provider} model listing request failed ({message})",
            status_code=502,
        ) from exc
    except error.URLError as exc:
        raise ModelCatalogError(
            f"Could not reach {provider} model endpoint ({exc.reason})",
            status_code=502,
        ) from exc

    if not raw_body.strip():
        return {}
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise ModelCatalogError(
            f"{provider} model listing returned invalid JSON",
            status_code=502,
        ) from exc

    if not isinstance(payload, dict):
        raise ModelCatalogError(
            f"{provider} model listing returned an unexpected payload",
            status_code=502,
        )
    return payload


def _list_openai_models(api_key: str) -> list[str]:
    payload = _request_json(
        provider="openai",
        url="https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ModelCatalogError("openai model listing did not include a data array", status_code=502)
    all_model_ids = [row.get("id") for row in rows if isinstance(row, dict)]
    ids = _dedupe_and_sort(model_id for model_id in all_model_ids if isinstance(model_id, str))
    preferred = [model_id for model_id in ids if model_id.startswith(("gpt-", "o1", "o3", "o4"))]
    return preferred or ids


def _list_google_models(api_key: str) -> list[str]:
    encoded_key = parse.urlencode({"key": api_key})
    payload = _request_json(
        provider="google",
        url=f"https://generativelanguage.googleapis.com/v1beta/models?{encoded_key}",
        headers={},
    )
    rows = payload.get("models")
    if not isinstance(rows, list):
        raise ModelCatalogError("google model listing did not include a models array", status_code=502)

    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        methods = row.get("supportedGenerationMethods")
        if isinstance(methods, list) and methods and "generateContent" not in methods:
            continue
        raw_name = row.get("name")
        if not isinstance(raw_name, str):
            continue
        model_id = raw_name.split("/", 1)[1] if raw_name.startswith("models/") else raw_name
        model_id = model_id.strip()
        if model_id:
            ids.append(model_id)

    deduped = _dedupe_and_sort(ids)
    preferred = [model_id for model_id in deduped if model_id.startswith("gemini-")]
    return preferred or deduped


def _list_anthropic_models(api_key: str) -> list[str]:
    payload = _request_json(
        provider="anthropic",
        url="https://api.anthropic.com/v1/models",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ModelCatalogError("anthropic model listing did not include a data array", status_code=502)
    all_model_ids = [row.get("id") for row in rows if isinstance(row, dict)]
    ids = _dedupe_and_sort(model_id for model_id in all_model_ids if isinstance(model_id, str))
    preferred = [model_id for model_id in ids if model_id.startswith("claude-")]
    return preferred or ids


def _list_zai_models(api_key: str) -> list[str]:
    payload = _request_json(
        provider="zai",
        url=f"{ZAI_OPENAI_BASE_URL}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ModelCatalogError("zai model listing did not include a data array", status_code=502)
    all_model_ids = [row.get("id") for row in rows if isinstance(row, dict)]
    ids = _dedupe_and_sort(model_id for model_id in all_model_ids if isinstance(model_id, str))
    preferred = [model_id for model_id in ids if model_id.startswith("glm-")]
    return preferred or ids


def list_models_for_provider(provider: str, api_key: str) -> list[str]:
    normalized_provider = (provider or "").strip().lower()
    normalized_key = (api_key or "").strip()
    if normalized_provider not in SUPPORTED_AI_PROVIDERS:
        raise ModelCatalogError(f"Unsupported AI provider: {provider}", status_code=400)
    if not normalized_key:
        raise ModelCatalogError("api key is required to fetch models", status_code=400)

    if normalized_provider == "openai":
        models = _list_openai_models(normalized_key)
    elif normalized_provider == "google":
        models = _list_google_models(normalized_key)
    elif normalized_provider == "anthropic":
        models = _list_anthropic_models(normalized_key)
    else:
        models = _list_zai_models(normalized_key)

    if not models:
        raise ModelCatalogError(f"No models returned for {normalized_provider}", status_code=502)
    return models
