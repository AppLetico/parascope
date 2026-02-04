from __future__ import annotations

from unittest.mock import patch
from pathlib import Path

from prscope.prd import get_prd_dir, get_prscope_dir


def test_get_prd_dir_uses_hyphen_separator(tmp_path):
    with patch("prscope.prd.get_prscope_dir", return_value=tmp_path):
        prd_dir = get_prd_dir("openclaw/openclaw")
        
        # Should use hyphen, not underscore
        assert prd_dir.name == "openclaw-openclaw"
        assert "_" not in prd_dir.name


def test_get_prd_dir_creates_directory(tmp_path):
    with patch("prscope.prd.get_prscope_dir", return_value=tmp_path):
        prd_dir = get_prd_dir("org/repo")
        
        assert prd_dir.exists()
        assert prd_dir.is_dir()


def test_get_prd_dir_without_repo_name(tmp_path):
    with patch("prscope.prd.get_prscope_dir", return_value=tmp_path):
        prd_dir = get_prd_dir()
        
        assert prd_dir == tmp_path / "prd"
        assert prd_dir.exists()


def test_get_prd_dir_handles_simple_repo_name(tmp_path):
    with patch("prscope.prd.get_prscope_dir", return_value=tmp_path):
        prd_dir = get_prd_dir("myorg/myrepo")
        
        assert prd_dir.name == "myorg-myrepo"
