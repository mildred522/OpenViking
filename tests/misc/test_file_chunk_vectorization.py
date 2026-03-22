# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for long-file chunked vectorization."""

from types import SimpleNamespace
from typing import Any

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.utils.embedding_utils import vectorize_file
from openviking_cli.session.user_id import UserIdentifier


class DummyEmbeddingQueue:
    """Collect embedding messages sent by vectorize_file()."""

    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def enqueue(self, msg) -> None:
        self.messages.append(msg)


class DummyQueueManager:
    """Minimal queue manager stub for embedding tests."""

    EMBEDDING = "embedding"

    def __init__(self) -> None:
        self.queue = DummyEmbeddingQueue()

    def get_queue(self, name):
        assert name == self.EMBEDDING
        return self.queue


class DummyVikingFS:
    """Minimal filesystem stub returning a fixed file body."""

    def __init__(self, content: str) -> None:
        self.content = content

    async def read_file(self, file_path: str, ctx=None) -> str:
        return self.content


@pytest.mark.asyncio
async def test_vectorize_file_chunks_long_text_and_preserves_base_summary(monkeypatch):
    queue_manager = DummyQueueManager()
    viking_fs = DummyVikingFS("ABCDEFGHIJ1234567890KLMNOPQRST")
    config = SimpleNamespace(file_chunk_chars=10, file_chunk_overlap=2)
    ctx = RequestContext(user=UserIdentifier("acc1", "user1", "agent1"), role=Role.USER)

    monkeypatch.setattr("openviking.utils.embedding_utils.get_queue_manager", lambda: queue_manager)
    monkeypatch.setattr("openviking.utils.embedding_utils.get_viking_fs", lambda: viking_fs)
    monkeypatch.setattr("openviking.utils.embedding_utils.get_openviking_config", lambda: config)

    await vectorize_file(
        file_path="viking://resources/demo/long.md",
        summary_dict={"name": "long.md", "summary": "long file summary"},
        parent_uri="viking://resources/demo",
        ctx=ctx,
    )

    messages = queue_manager.queue.messages
    assert len(messages) == 5

    assert messages[0].context_data["uri"] == "viking://resources/demo/long.md"
    assert messages[0].message == "long file summary"

    chunk_uris = [msg.context_data["uri"] for msg in messages[1:]]
    assert chunk_uris == [
        "viking://resources/demo/long.md#chunk_0000",
        "viking://resources/demo/long.md#chunk_0001",
        "viking://resources/demo/long.md#chunk_0002",
        "viking://resources/demo/long.md#chunk_0003",
    ]
    assert [msg.message for msg in messages[1:]] == [
        "ABCDEFGHIJ",
        "IJ12345678",
        "7890KLMNOP",
        "OPQRST",
    ]
    assert all(msg.context_data["parent_uri"] == "viking://resources/demo" for msg in messages[1:])
    assert messages[1].context_data["meta"]["chunk_index"] == 0
    assert messages[1].context_data["meta"]["chunk_count"] == 4
    assert messages[1].context_data["meta"]["source_uri"] == "viking://resources/demo/long.md"
