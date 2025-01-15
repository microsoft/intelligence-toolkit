# Developing 

## Requirements

- Python 3.11 ([Download](https://www.python.org/downloads/))
- poetry < 2.0.0 ([Download](https://python-poetry.org/docs/#installing-with-the-official-installer))
- wkhtmltopdf (used to generate PDF reports)

    - Windows: ([Download](https://wkhtmltopdf.org/downloads.html))

    - Linux:  `sudo apt-get install wkhtmltopdf`

    - MacOS: `brew install homebrew/cask/wkhtmltopdf`


## Running the app

## GPT settings

You can configure your OpenAI access when running the app via `Settings page`, or using environment variables.

#### Default values: 
```
OPENAI_API_MODEL="gpt-4o-mini"
OPENAI_TYPE="OpenAI" ## Other option available: Azure OpenAI
AZURE_AUTH_TYPE="Azure Key" # if OPENAI_TYPE==Azure OpenAI
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
```

### OpenAI
OPENAI_API_KEY=<OPENAI_API_KEY>

### Azure OpenAI
```
OPENAI_TYPE="Azure OpenAI"
AZURE_OPENAI_VERSION=2023-12-01-preview
AZURE_OPENAI_ENDPOINT="https://<ENDPOINT>.azure.com/"
OPENAI_API_KEY=<AZURE_OPENAI_API_KEY>

#If Azure OpenAI using Managed Identity:
AZURE_AUTH_TYPE="Managed Identity"
```

### Running locally

Windows: Search and open the app `Windows Powershell` on Windows start menu

Linux and Mac: Open `Terminal`

For any OS:

Navigate to the folder where you cloned this repo. 

Use `cd `+ the path to the folder. For example:

`cd C:\Users\user01\projects\intelligence-toolkit`

Run `poetry install` and wait for the packages installation.

#### Run the app:

Run `poetry run poe run_streamlit`, and it will automatically open the app in your default browser in `localhost:8081`

#### Use the API

You can also replicate the examples in your own environment running `pip install intelligence-toolkit`.

See the documentation and an example of how to run the code with your data to obtain results without the need to run the UI.
- [Anonymize Case Data](./app/workflows/anonymize_case_data/README.md)

    - [Example](./example_notebooks/anonymize_case_data.ipynb)

- [Compare Case Groups](./app/workflows/compare_case_groups/README.md)

    - [Example](./example_notebooks/compare_case_groups.ipynb)

- [Detect Case Patterns](./app/workflows/detect_case_patterns/README.md)

    - [Example](./example_notebooks/detect_case_patterns.ipynb)

- [Detect Entity Networks](./app/workflows/detect_entity_networks/README.md)

    - [Example](./example_notebooks/detect_entity_networks.ipynb)

- [Extract Record Data](./app/workflows/extract_record_data/README.md)

    - [Example](./example_notebooks/extract_record_data.ipynb)

- [Generate Mock Data](./app/workflows/generate_mock_data/README.md)

    - [Example](./example_notebooks/generate_mock_data.ipynb)

- [Match Entity Records](./app/workflows/match_entity_records/README.md)

    - [Example](./example_notebooks/match_entity_records.ipynb)
    
- [Query Text Data](./app/workflows/query_text_data/README.md)

    - [Example](./example_notebooks/query_text_data.ipynb)


### Running with docker

##### Recommended configuration:

- *Minimum disk space*: 8GB 
- *Minimum memory*: 4GB

Download, install and then open docker app: https://www.docker.com/products/docker-desktop/

Then, open a terminal:
Windows: Search and open the app `Windows Powershell` on Windows start menu

Linux and Mac: Open `Terminal`

For any OS:

Navigate to the folder where you cloned this repo. 

Use `cd `+ the path to the folder. For example:

`cd C:\Users\user01\projects\intelligence-toolkit`

Build the container:

`docker build . -t intelligence-toolkit`

Once the build is finished, run the docker container:

- via terminal:

    `docker run -d --name intelligence-toolkit -p 80:80 intelligence-toolkit`

Open [localhost:80](http://localhost:80)

  **Note that docker might sleep and you might need to start it again. Open Docker Desktop, in the left menu click on Container and press play on intelligence-toolkit.**

# Lifecycle Scripts

For Lifecycle scripts it utilizes [Poetry](https://python-poetry.org/docs#installation) and [poethepoet](https://pypi.org/project/poethepoet/) to manage build scripts.

Available scripts are:

- `poetry run poe test_unit` - This will execute unit tests on api.
- `poetry run poe test_smoke` - This will execute smoke tests on api.
- `poetry run poe check` - This will perform a suite of static checks across the package, including:
  - formatting
  - documentation formatting
  - linting
  - security patterns
  - type-checking
- `poetry run poe fix` - This will apply any available auto-fixes to the package. Usually this is just formatting fixes.
- `poetry run poe fix_unsafe` - This will apply any available auto-fixes to the package, including those that may be unsafe.
- `poetry run poe format` - Explicitly run the formatter across the package.

