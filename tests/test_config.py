import pytest

from legal_triage.config import ConfigurationError, validate_local_model_name, validate_local_ollama_url


def test_local_loopback_urls_are_allowed():
    assert validate_local_ollama_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert validate_local_ollama_url("http://localhost:11434/") == "http://localhost:11434"


@pytest.mark.parametrize("url", [
    "https://api.example.com",
    "http://192.168.1.12:11434",
    "http://0.0.0.0:11434",
    "http://127.0.0.1:11434/private",
    "http://user:password@127.0.0.1:11434",
])
def test_remote_or_non_root_ollama_urls_are_rejected(url):
    with pytest.raises(ConfigurationError):
        validate_local_ollama_url(url)


def test_cloud_model_tags_are_rejected():
    with pytest.raises(ConfigurationError):
        validate_local_model_name("gemma4:cloud")
    assert validate_local_model_name("gemma4:e2b") == "gemma4:e2b"


def test_local_ollama_client_does_not_inherit_proxies():
    from legal_triage.local_runtime import LocalOllama
    runtime = LocalOllama()
    assert runtime.session.trust_env is False


def test_installed_ollama_model_alias_latest_is_normalized():
    from legal_triage.local_runtime import LocalOllama

    runtime = LocalOllama()
    runtime._request = lambda method, endpoint, payload=None: {  # type: ignore[method-assign]
        "models": [{"name": "nomic-embed-text:latest"}, {"name": "gemma4:e2b"}]
    }
    names = runtime.installed_models()
    assert "nomic-embed-text" in names
    assert "nomic-embed-text:latest" in names
    assert "gemma4:e2b" in names
