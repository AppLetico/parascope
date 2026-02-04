from __future__ import annotations

from prscope.store import Store


def test_evaluation_deduplication(tmp_path):
    db_path = tmp_path / "prscope.db"
    store = Store(db_path=db_path)

    repo = store.upsert_upstream_repo("owner/repo")
    pr = store.upsert_pull_request(
        repo_id=repo.id,
        number=1,
        state="closed",
        title="Test PR",
        body="",
        author="alice",
        labels=["test"],
        updated_at="2024-01-01T00:00:00Z",
        merged_at="2024-01-02T00:00:00Z",
        head_sha="sha1",
        html_url="https://example.com/pr/1",
    )

    local_profile_sha = "local-sha"
    pr_head_sha = "sha1"

    assert store.evaluation_exists(pr.id, local_profile_sha, pr_head_sha) is False

    store.save_evaluation(
        pr_id=pr.id,
        local_profile_sha=local_profile_sha,
        pr_head_sha=pr_head_sha,
        rule_score=0.8,
        final_score=0.8,
        matched_features=["security"],
        signals={"file_count": 1},
        llm_result=None,
        decision="relevant",
    )

    assert store.evaluation_exists(pr.id, local_profile_sha, pr_head_sha) is True
