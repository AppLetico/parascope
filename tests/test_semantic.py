from __future__ import annotations

from pathlib import Path

from parascope.semantic import (
    CodeChunk,
    read_local_files,
    extract_matching_files,
    SKIP_DIRS,
    CODE_EXTENSIONS,
)


def test_code_chunk_summary():
    chunk = CodeChunk(
        path="src/auth/handler.py",
        content="def authenticate(user):\n    return True",
        start_line=1,
        end_line=2,
    )
    summary = chunk.summary()
    assert "src/auth/handler.py" in summary
    assert "def authenticate" in summary


def test_skip_dirs():
    assert ".git" in SKIP_DIRS
    assert "node_modules" in SKIP_DIRS
    assert "__pycache__" in SKIP_DIRS


def test_code_extensions():
    assert ".py" in CODE_EXTENSIONS
    assert ".ts" in CODE_EXTENSIONS
    assert ".tsx" in CODE_EXTENSIONS


def test_read_local_files(tmp_path):
    # Create test files
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hello')")
    (src / "utils.ts").write_text("export const foo = 1;")
    
    # Create ignored directory
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "pkg.js").write_text("ignored")
    
    chunks = read_local_files(tmp_path)
    
    paths = [c.path for c in chunks]
    assert "src/main.py" in paths
    assert "src/utils.ts" in paths
    # node_modules should be skipped
    assert not any("node_modules" in p for p in paths)


def test_extract_matching_files(tmp_path):
    # Create test files
    auth = tmp_path / "src" / "auth"
    auth.mkdir(parents=True)
    (auth / "login.py").write_text("def login(): pass")
    (auth / "logout.py").write_text("def logout(): pass")
    
    api = tmp_path / "src" / "api"
    api.mkdir(parents=True)
    (api / "users.py").write_text("def get_users(): pass")
    
    # Match by feature paths
    matched = extract_matching_files(
        repo_root=tmp_path,
        pr_files=["something/login.py"],  # Should match by filename
        feature_paths=["**/auth/**"],
    )
    
    paths = [c.path for c in matched]
    assert any("login.py" in p for p in paths)
