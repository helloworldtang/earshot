"""deliver 模块单测:exomind 命令 wire 契约 + 退出码分类 + webhook。"""

from pathlib import Path

from earshot.deliver import build_exomind_args, deliver_exomind, deliver_webhook


class _FakeCompleted:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_build_exomind_args_wire() -> None:
    md = Path("/tmp/note.md")
    args = build_exomind_args(md, "标题", ["旁听", "B站"])
    assert args == [
        "exomind",
        "ingest",
        "--file",
        "/tmp/note.md",
        "--origin",
        "earshot",
        "--title",
        "标题",
        "--tag",
        "旁听",
        "--tag",
        "B站",
    ]


def test_build_exomind_args_without_title() -> None:
    args = build_exomind_args(Path("/tmp/a.md"), "", [])
    assert "--title" not in args
    assert "--tag" not in args


def test_deliver_exomind_success() -> None:
    def runner(_cmd: list[str], **_kw: object) -> _FakeCompleted:
        return _FakeCompleted(0, stdout="job done\nentities=5")

    result = deliver_exomind(Path("/tmp/a.md"), "t", [], runner=runner)  # type: ignore[arg-type]
    assert result.status == "delivered"
    assert result.detail == "entities=5"


def test_deliver_exomind_pending_exit_3() -> None:
    def runner(_cmd: list[str], **_kw: object) -> _FakeCompleted:
        return _FakeCompleted(3, stdout="")

    result = deliver_exomind(Path("/tmp/a.md"), "t", [], runner=runner)  # type: ignore[arg-type]
    assert result.status == "pending"


def test_deliver_exomind_failure() -> None:
    def runner(_cmd: list[str], **_kw: object) -> _FakeCompleted:
        return _FakeCompleted(1, stderr="boom at line 2\nquota exceeded")

    result = deliver_exomind(Path("/tmp/a.md"), "t", [], runner=runner)  # type: ignore[arg-type]
    assert result.status == "failed"
    assert result.detail == "quota exceeded"


def test_deliver_exomind_cli_missing() -> None:
    def runner(_cmd: list[str], **_kw: object) -> _FakeCompleted:
        raise FileNotFoundError("exomind")

    result = deliver_exomind(Path("/tmp/a.md"), "t", [], runner=runner)  # type: ignore[arg-type]
    assert result.status == "failed"
    assert "exomind CLI" in result.detail


def test_deliver_webhook_posts_payload() -> None:
    seen: dict[str, object] = {}

    def poster(url: str, payload: dict[str, str | list[str]]) -> None:
        seen["url"] = url
        seen["payload"] = payload

    result = deliver_webhook("https://example.com/hook", "标题", "# 正文", ["a"], poster=poster)
    assert result.status == "delivered"
    assert seen["url"] == "https://example.com/hook"
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["title"] == "标题"
    assert payload["tags"] == ["a"]


def test_deliver_webhook_failure() -> None:
    def poster(_url: str, _payload: dict[str, str | list[str]]) -> None:
        raise RuntimeError("connection refused")

    result = deliver_webhook("https://example.com/hook", "t", "c", [], poster=poster)
    assert result.status == "failed"
    assert "connection refused" in result.detail
