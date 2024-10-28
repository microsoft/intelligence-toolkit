# Copyright (c) 2024 Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#


from toolkit.helpers.texts import clean_for_column_name, clean_text_for_csv


class TestCleanTextForColumnName:
    def test_clean_text_for_csv_empty(self) -> None:
        assert clean_for_column_name("") == ""

    def test_clean_text_for_csv_alpha_numeric(self) -> None:
        assert clean_for_column_name("Hello123") == "Hello123"

    def test_clean_text_for_csv_with_spaces(self) -> None:
        assert clean_for_column_name("Hello World 123") == "Hello World 123"

    def test_clean_text_for_csv_special_characters(self) -> None:
        assert clean_for_column_name("Hello, World! 123") == "Hello World 123"

    def test_clean_text_for_csv_email(self) -> None:
        assert clean_for_column_name("user@example.com") == "userexamplecom"

    def test_clean_text_for_csv_ampersand(self) -> None:
        assert clean_for_column_name("R&D") == "R&D"

    def test_clean_text_for_csv_plus_sign(self) -> None:
        assert clean_for_column_name("C++") == "C++"

    def test_clean_text_for_csv_at_symbol(self) -> None:
        assert clean_for_column_name("user@domain") == "userdomain"

    def test_clean_text_for_csv_numbers(self) -> None:
        assert clean_for_column_name(123456) == "123456"

    def test_clean_text_for_csv_mixed_characters(self) -> None:
        assert clean_for_column_name("Hello@World&123+") == "HelloWorld&123+"

    def test_clean_text_for_csv_only_special_characters(self) -> None:
        assert clean_for_column_name("!@#$%^&*()_+={}-[]:\";'<>?,./") == "&()_+-"

    def test_clean_text_for_csv_unicode_characters(self) -> None:
        assert clean_for_column_name("你好,世界") == "你好世界"

    def test_clean_text_for_csv_underscore(self) -> None:
        assert clean_for_column_name("file_name") == "file_name"

    def test_clean_text_for_csv_dash(self) -> None:
        assert clean_for_column_name("file-name") == "file-name"


class TestCleanTextForCsv:
    def test_clean_text_for_csv_empty(self) -> None:
        assert clean_text_for_csv("") == ""

    def test_clean_text_for_csv_alpha_numeric(self) -> None:
        assert clean_text_for_csv("Hello123") == "Hello123"

    def test_clean_text_for_csv_with_spaces(self) -> None:
        assert clean_text_for_csv("Hello World 123") == "Hello World 123"

    def test_clean_text_for_csv_special_characters(self) -> None:
        assert clean_text_for_csv("Hello, World! 123") == "Hello World 123"

    def test_clean_text_for_csv_email(self) -> None:
        assert clean_text_for_csv("user@example.com") == "user@examplecom"

    def test_clean_text_for_csv_ampersand(self) -> None:
        assert clean_text_for_csv("R&D") == "R&D"

    def test_clean_text_for_csv_plus_sign(self) -> None:
        assert clean_text_for_csv("C++") == "C++"

    def test_clean_text_for_csv_at_symbol(self) -> None:
        assert clean_text_for_csv("user@domain") == "user@domain"

    def test_clean_text_for_csv_numbers(self) -> None:
        assert clean_text_for_csv(123456) == "123456"

    def test_clean_text_for_csv_mixed_characters(self) -> None:
        assert clean_text_for_csv("Hello@World&123+") == "Hello@World&123+"

    def test_clean_text_for_csv_only_special_characters(self) -> None:
        assert clean_text_for_csv("!@#$%^&*()_+={}[]:\";'<>?,./") == "@&_+"

    def test_clean_text_for_csv_unicode_characters(self) -> None:
        assert clean_text_for_csv("你好,世界") == "你好世界"

    def test_clean_text_for_csv_underscore(self) -> None:
        assert clean_text_for_csv("file_name") == "file_name"

    def test_clean_text_for_csv_dash(self) -> None:
        assert clean_text_for_csv("file-name") == "filename"
