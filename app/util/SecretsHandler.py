import os

from httpx import delete

class SecretsHandler:
    _directory = ".streamlit"
    _file_name = "app_secrets.toml"
    _file_path = os.path.join(_directory, _file_name)

    def __init__(self):
        if not os.path.exists(self._directory):
            os.makedirs(self._directory)
        if not os.path.exists(self._file_path):
            with(open(self._file_path, "w+")) as f:
                f.write("")

    def read_values_from_file(self):
        values = {}
        with open(self._file_path, 'r') as file:
            for line in file:
                key_value_pairs = line.strip().split(';')
                for pair in key_value_pairs:
                    if pair == '': continue
                    key, value = pair.split(':')
                    values[key] = value
        return values
    
    def get_secret(self, value):
        values = self.read_values_from_file()
        for key in self.read_values_from_file().keys():
            if key == value:
                return values[key]
        return ''
    
    def delete_secret(self, key):
        values = self.read_values_from_file()
        if key in values:
            del values[key]
            self.write_values_to_file(values)

    def write_values_to_file(self, values):
        with open(self._file_path, 'w') as file:
            for key, value in values.items():
                file.write(f"{key}:{value};\n")

    def update_values(self, updates):
        existing_values = self.read_values_from_file(self._file_path)
        for key, value in updates.items():
            existing_values[key] = value
        self.write_values_to_file(self._file_path, existing_values)
    
    def write_secret(self, key, value):
        values = self.read_values_from_file()
        values[key] = value
        self.write_values_to_file(values)