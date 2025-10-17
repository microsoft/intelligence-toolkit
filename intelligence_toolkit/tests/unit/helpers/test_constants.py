# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import os

from intelligence_toolkit.helpers import constants


def test_pdf_encoding():
    assert constants.PDF_ENCODING == "UTF-8"


def test_pdf_margin_inches():
    assert constants.PDF_MARGIN_INCHES == 0.75
    assert isinstance(constants.PDF_MARGIN_INCHES, float)


def test_attribute_value_separator():
    assert constants.ATTRIBUTE_VALUE_SEPARATOR == "=="


def test_cache_path_exists():
    assert constants.CACHE_PATH is not None
    assert isinstance(constants.CACHE_PATH, str)
    assert os.path.exists(constants.CACHE_PATH)


def test_cache_path_contains_intelligence_toolkit():
    assert "intelligence-toolkit-data" in constants.CACHE_PATH
    assert "cache" in constants.CACHE_PATH


def test_vector_store_max_retries():
    assert constants.VECTOR_STORE_MAX_RETRIES == 5
    assert isinstance(constants.VECTOR_STORE_MAX_RETRIES, int)


def test_vector_store_max_retries_wait_time():
    assert constants.VECTOR_STORE_MAX_RETRIES_WAIT_TIME == 1
    assert isinstance(constants.VECTOR_STORE_MAX_RETRIES_WAIT_TIME, int)
