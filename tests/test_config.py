from __future__ import annotations

from unittest.mock import patch

from parascope.config import ParascopeConfig


def test_config_load_local_repo_and_defaults(tmp_path):
    config_path = tmp_path / "parascope.yml"
    config_path.write_text(
        """
local_repo: ./local
sync:
  state: open
  max_prs: 5
  fetch_files: false
llm:
  enabled: true
  model: claude-3-opus
  temperature: 0.2
  max_tokens: 1500
        """.strip()
    )

    config = ParascopeConfig.load(tmp_path)

    assert config.local_repo == "./local"
    
    # Mock get_repo_root to return tmp_path for relative path resolution
    with patch("parascope.config.get_repo_root", return_value=tmp_path):
        assert config.get_local_repo_path() == (tmp_path / "local").resolve()
    
    assert config.sync.state == "open"
    assert config.sync.max_prs == 5
    assert config.sync.fetch_files is False
    assert config.llm.enabled is True
    assert config.llm.model == "claude-3-opus"
    assert config.llm.temperature == 0.2
    assert config.llm.max_tokens == 1500
