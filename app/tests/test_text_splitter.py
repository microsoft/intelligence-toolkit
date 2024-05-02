import pytest

from app.AI.text_splitter import TextSplitter


@pytest.fixture
def text_splitter():
    return TextSplitter(20)

def test_text_splitter(text_splitter):
    text = "This is a test string for text splitting."
    expected_chunks = [
        "This is a test",
        "string for text",
        "splitting."
    ]
    chunks = text_splitter.split(text)
    assert chunks == expected_chunks

def test_text_splitter_empty_input(text_splitter):
    text = ""
    chunks = text_splitter.split(text)
    assert chunks == []

def test_text_splitter_no_overlap():
    splitter = TextSplitter(20, 5)
    text = "This is a test string for text splitting."
    expected_chunks = [
        "This is a test",
        "test string for",
        "for text splitting."
    ]
    chunks = splitter.split(text)
    assert chunks == expected_chunks