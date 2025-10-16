# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from intelligence_toolkit.helpers.document_processor import convert_files_to_chunks


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_convert_files_to_chunks_txt_file(temp_dir):
    # Create a test text file
    txt_file = Path(temp_dir) / "test.txt"
    txt_file.write_text("This is a test document with some content.")
    
    result = convert_files_to_chunks([str(txt_file)], chunk_size=5)
    
    assert "test.txt" in result
    assert len(result["test.txt"]) > 0


def test_convert_files_to_chunks_csv_file(temp_dir):
    # Create a test CSV file
    csv_file = Path(temp_dir) / "test.csv"
    df = pd.DataFrame({
        "col1": ["value1", "value2"],
        "col2": ["value3", "value4"]
    })
    df.to_csv(csv_file, index=False)
    
    result = convert_files_to_chunks([str(csv_file)], chunk_size=50)
    
    # Should create chunks for each row
    assert "test.csv_1" in result
    assert "test.csv_2" in result


def test_convert_files_to_chunks_json_list(temp_dir):
    # Create a test JSON file with a list
    json_file = Path(temp_dir) / "test.json"
    json_data = [
        {"key1": "value1"},
        {"key2": "value2"}
    ]
    json_file.write_text(json.dumps(json_data))
    
    result = convert_files_to_chunks([str(json_file)], chunk_size=50)
    
    assert "test.json_1" in result
    assert "test.json_2" in result


def test_convert_files_to_chunks_json_object(temp_dir):
    # Create a test JSON file with a single object
    json_file = Path(temp_dir) / "test.json"
    json_data = {"key": "value", "nested": {"inner": "data"}}
    json_file.write_text(json.dumps(json_data))
    
    result = convert_files_to_chunks([str(json_file)], chunk_size=50)
    
    assert "test.json" in result


def test_convert_files_to_chunks_pdf_file(temp_dir):
    # Mock PDF reading since creating real PDFs is complex
    pdf_file = Path(temp_dir) / "test.pdf"
    pdf_file.write_text("dummy pdf content")
    
    with patch("intelligence_toolkit.helpers.document_processor.PdfReader") as mock_reader:
        mock_pdf = MagicMock()
        mock_pdf.get_num_pages.return_value = 2
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_reader.return_value = mock_pdf
        
        result = convert_files_to_chunks([str(pdf_file)], chunk_size=50)
        
        assert "test.pdf" in result


def test_convert_files_to_chunks_multiple_files(temp_dir):
    # Create multiple test files
    txt_file = Path(temp_dir) / "test1.txt"
    txt_file.write_text("Content of file 1")
    
    txt_file2 = Path(temp_dir) / "test2.txt"
    txt_file2.write_text("Content of file 2")
    
    result = convert_files_to_chunks([str(txt_file), str(txt_file2)], chunk_size=50)
    
    assert "test1.txt" in result
    assert "test2.txt" in result


def test_convert_files_to_chunks_with_callbacks(temp_dir):
    txt_file = Path(temp_dir) / "test.txt"
    txt_file.write_text("Test content")
    
    callback = MagicMock()
    callback.on_batch_change = MagicMock()
    
    result = convert_files_to_chunks([str(txt_file)], chunk_size=50, callbacks=[callback])
    
    callback.on_batch_change.assert_called()


def test_convert_files_to_chunks_filename_sanitization(temp_dir):
    # Create file with special characters in name
    txt_file = Path(temp_dir) / "test (with spaces).txt"
    txt_file.write_text("Test content")
    
    result = convert_files_to_chunks([str(txt_file)], chunk_size=50)
    
    # Parentheses and spaces should be removed/replaced
    assert "test_with_spaces.txt" in result


def test_convert_files_to_chunks_chunk_structure(temp_dir):
    txt_file = Path(temp_dir) / "test.txt"
    txt_file.write_text("Short text")
    
    result = convert_files_to_chunks([str(txt_file)], chunk_size=50)
    
    # Verify chunk structure
    chunk_json = json.loads(result["test.txt"][0])
    assert "title" in chunk_json
    assert "text_chunk" in chunk_json
    assert "chunk_id" in chunk_json
    assert chunk_json["chunk_id"] == 1


def test_convert_files_to_chunks_empty_list():
    result = convert_files_to_chunks([], chunk_size=50)
    
    assert len(result) == 0


def test_convert_files_to_chunks_csv_row_format(temp_dir):
    csv_file = Path(temp_dir) / "test.csv"
    df = pd.DataFrame({
        "name": ["Alice"],
        "age": [30]
    })
    df.to_csv(csv_file, index=False)
    
    result = convert_files_to_chunks([str(csv_file)], chunk_size=100)
    
    # Check that the chunk contains the formatted row data
    chunk_json = json.loads(result["test.csv_1"][0])
    assert "name: Alice" in chunk_json["text_chunk"]
    assert "age: 30" in chunk_json["text_chunk"]
