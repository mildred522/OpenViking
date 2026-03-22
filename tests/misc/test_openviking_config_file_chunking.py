# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for file chunking settings in OpenVikingConfig."""

import pytest

from openviking_cli.utils.config.open_viking_config import OpenVikingConfig


def _minimal_config_dict() -> dict:
    return {
        "embedding": {
            "dense": {
                "provider": "openai",
                "api_key": "test-key",
                "model": "text-embedding-3-small",
                "dimension": 1536,
            }
        }
    }


def test_openviking_config_file_chunk_defaults():
    config = OpenVikingConfig.from_dict(_minimal_config_dict())

    assert config.file_chunk_chars == 4000
    assert config.file_chunk_overlap == 400


def test_openviking_config_accepts_custom_file_chunk_settings():
    config = OpenVikingConfig.from_dict(
        {
            **_minimal_config_dict(),
            "file_chunk_chars": 2048,
            "file_chunk_overlap": 256,
        }
    )

    assert config.file_chunk_chars == 2048
    assert config.file_chunk_overlap == 256


def test_openviking_config_rejects_invalid_file_chunk_overlap():
    with pytest.raises(ValueError, match="file_chunk_overlap must be smaller than file_chunk_chars"):
        OpenVikingConfig.from_dict(
            {
                **_minimal_config_dict(),
                "file_chunk_chars": 512,
                "file_chunk_overlap": 512,
            }
        )
