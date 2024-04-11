import os

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

    def write_secret(self, key, value):
        with(open(self._file_path, "w")) as f:
            f.write(f"{key}:{value};")

    def get_secret(self, key) -> str:
        with(open(self._file_path, "r")) as f:
            secrets = f.read()
            secret_key = secrets.split(";")
            for key_value in secret_key:
                secret_key = key_value.split(":")[0]
                if key == secret_key:
                    return key_value.split(":")[1]
        return ''