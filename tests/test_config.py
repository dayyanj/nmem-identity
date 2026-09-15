"""Config precedence: defaults < TOML < env, with type coercion (pydantic Settings)."""
from nmem_identity import config
from nmem_identity.config import Settings

FIELDS = list(Settings.model_fields)


def _clear_env(monkeypatch):
    for f in FIELDS:
        monkeypatch.delenv("NMEM_IDENTITY_" + f.upper(), raising=False)
    monkeypatch.delenv("NMEM_IDENTITY_CONFIG", raising=False)


def test_defaults(monkeypatch):
    _clear_env(monkeypatch)
    s = Settings()
    assert s.dsn.startswith("postgresql://")
    assert isinstance(s.matcher_port, int)
    assert isinstance(s.face_det_size, int)
    assert s.face_model_pack == "buffalo_s"


def test_env_overrides_and_coerces(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("NMEM_IDENTITY_DSN", "postgresql://x/y")
    monkeypatch.setenv("NMEM_IDENTITY_FACE_DET_SIZE", "320")
    s = Settings()
    assert s.dsn == "postgresql://x/y"
    assert s.face_det_size == 320 and isinstance(s.face_det_size, int)


def test_renamed_vars_resolve_to_canonical(monkeypatch):
    # canonical names after rename (were FACE_PORT / VOICE_PORT / FACE_PACK / FACE_TAU)
    _clear_env(monkeypatch)
    monkeypatch.setenv("NMEM_IDENTITY_FACE_EMBED_PORT", "9500")
    monkeypatch.setenv("NMEM_IDENTITY_VOICE_EMBED_PORT", "9600")
    monkeypatch.setenv("NMEM_IDENTITY_FACE_MODEL_PACK", "buffalo_l")
    monkeypatch.setenv("NMEM_IDENTITY_FACE_TAU_MATCH", "0.42")
    s = Settings()
    assert s.face_embed_port == 9500
    assert s.voice_embed_port == 9600
    assert s.face_model_pack == "buffalo_l"
    assert s.face_tau_match == 0.42


def test_toml_sits_between_defaults_and_env(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    f = tmp_path / "nmem-identity.toml"
    f.write_text('matcher_port = 9999\ndsn = "postgresql://toml/db"\n')
    monkeypatch.setenv("NMEM_IDENTITY_CONFIG", str(f))
    s = Settings()
    assert s.matcher_port == 9999 and s.dsn == "postgresql://toml/db"

    # env still wins over the TOML value
    monkeypatch.setenv("NMEM_IDENTITY_DSN", "postgresql://env/db")
    assert Settings().dsn == "postgresql://env/db"


def test_toml_accepts_nmem_identity_section(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    f = tmp_path / "nmem-identity.toml"
    f.write_text('[nmem_identity]\nmatcher_port = 8888\n')
    monkeypatch.setenv("NMEM_IDENTITY_CONFIG", str(f))
    assert Settings().matcher_port == 8888


def test_accessor_shims_delegate_to_singleton():
    # get()/dsn()/all_values() are thin views over the module singleton.
    assert config.dsn() == config.settings.dsn
    assert config.dsn().startswith("postgresql://")
    assert config.get("matcher_port") == config.settings.matcher_port
    assert config.all_values()["face_model_pack"] == config.settings.face_model_pack
