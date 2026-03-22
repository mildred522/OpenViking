# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for chunk result collapse in hierarchical retrieval."""

from typing import Any, cast

import pytest

from openviking.retrieve.hierarchical_retriever import HierarchicalRetriever
from openviking.server.identity import RequestContext, Role
from openviking_cli.retrieve.types import ContextType, TypedQuery
from openviking_cli.session.user_id import UserIdentifier


class DummyStorage:
    """Minimal storage stub for retrieval collapse tests."""

    def __init__(self) -> None:
        self.collection_name = "context"

    async def collection_exists_bound(self) -> bool:
        return True

    async def search_global_roots_in_tenant(
        self,
        ctx,
        query_vector=None,
        sparse_query_vector=None,
        context_type=None,
        target_directories=None,
        extra_filter=None,
        limit: int = 10,
    ):
        return []

    async def search_children_in_tenant(
        self,
        ctx,
        parent_uri: str,
        query_vector=None,
        sparse_query_vector=None,
        context_type=None,
        target_directories=None,
        extra_filter=None,
        limit: int = 10,
    ):
        if parent_uri != "viking://resources/demo":
            return []
        return [
            {
                "uri": "viking://resources/demo/guide.md#chunk_0000",
                "context_type": "resource",
                "level": 2,
                "abstract": "",
                "_score": 0.95,
            },
            {
                "uri": "viking://resources/demo/guide.md#chunk_0001",
                "context_type": "resource",
                "level": 2,
                "abstract": "",
                "_score": 0.94,
            },
            {
                "uri": "viking://resources/demo/guide.md",
                "context_type": "resource",
                "level": 2,
                "abstract": "Guide summary",
                "_score": 0.75,
            },
            {
                "uri": "viking://resources/demo/faq.md",
                "context_type": "resource",
                "level": 2,
                "abstract": "FAQ summary",
                "_score": 0.80,
            },
        ]


@pytest.mark.asyncio
async def test_convert_to_matched_contexts_collapses_chunk_hits(monkeypatch):
    retriever = HierarchicalRetriever(
        storage=cast(Any, DummyStorage()), embedder=None, rerank_config=None
    )
    ctx = RequestContext(user=UserIdentifier("acc1", "user1", "agent1"), role=Role.USER)

    monkeypatch.setattr("openviking.retrieve.hierarchical_retriever.get_viking_fs", lambda: None)

    results = await retriever._convert_to_matched_contexts(
        [
            {
                "uri": "viking://resources/demo/guide.md#chunk_0000",
                "context_type": "resource",
                "level": 2,
                "abstract": "",
                "_final_score": 0.95,
            },
            {
                "uri": "viking://resources/demo/guide.md",
                "context_type": "resource",
                "level": 2,
                "abstract": "Guide summary",
                "_final_score": 0.75,
            },
            {
                "uri": "viking://resources/demo/faq.md",
                "context_type": "resource",
                "level": 2,
                "abstract": "FAQ summary",
                "_final_score": 0.80,
            },
        ],
        ctx=ctx,
    )

    assert [result.uri for result in results] == [
        "viking://resources/demo/guide.md",
        "viking://resources/demo/faq.md",
    ]
    assert results[0].abstract == "Guide summary"


@pytest.mark.asyncio
async def test_retrieve_returns_enough_unique_results_after_chunk_collapse(monkeypatch):
    retriever = HierarchicalRetriever(
        storage=cast(Any, DummyStorage()), embedder=None, rerank_config=None
    )
    ctx = RequestContext(user=UserIdentifier("acc1", "user1", "agent1"), role=Role.USER)

    monkeypatch.setattr("openviking.retrieve.hierarchical_retriever.get_viking_fs", lambda: None)

    result = await retriever.retrieve(
        TypedQuery(
            query="guide",
            context_type=ContextType.RESOURCE,
            intent="",
            target_directories=["viking://resources/demo"],
        ),
        ctx=ctx,
        limit=2,
    )

    assert [matched.uri for matched in result.matched_contexts] == [
        "viking://resources/demo/guide.md",
        "viking://resources/demo/faq.md",
    ]
