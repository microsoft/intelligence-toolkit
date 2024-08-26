# Developing 

## Requirements

- Python 3.11 ([Download](https://www.python.org/downloads/))
    - We haver available two dependency management:
        - poetry ([Download](https://python-poetry.org/docs/#installation))
        - pip
- wkhtmltopdf (used to generate PDF reports)

    - Windows: ([Download](https://wkhtmltopdf.org/downloads.html))

    - Linux:  `sudo apt-get install wkhtmltopdf`

    - MacOS: `brew install homebrew/cask/wkhtmltopdf`


## Install Dependencies

- Using poetry

    1. `poetry install`

- Using pip
    1. Set up a virtual environment:

        `python -m venv ./venv`

        or
        
        `python3 -m venv ./venv`
    2. Run the activate script: 

        `source venv/bin/activate`  (Linux)

        or 

        `.\venv\Scripts\Activate` (Windows with Powershell)

    3. `pip install -r requirements.txt`

### LLM API access

#### Default values: 
```
OPENAI_API_MODEL="gpt-4o-2024-08-06"
OPENAI_TYPE="OpenAI"
AZURE_AUTH_TYPE="Azure Key" # if OPENAI_TYPE==Azure OpenAI
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
```

### Azure OpenAI
```
OPENAI_TYPE="Azure OpenAI"
AZURE_OPENAI_VERSION=2023-12-01-preview
AZURE_OPENAI_ENDPOINT="https://<ENDPOINT>.azure.com/"
OPENAI_API_KEY=<OPENAI_API_KEY>
AZURE_AUTH_TYPE="Managed Identity" # if not default Azure Key
```

### OpenAI
```
OPENAI_API_KEY=<OPENAI_API_KEY>
```

## Running code-only 
- [Attribute Patterns](./toolkit/attribute_patterns/README.md)

    - [Example](./examples/attribute_patterns.ipynb): See an example of how to run the code with your data to obtain results without the need to run the UI.

- [Question Answering](./toolkit/question_answering/README.md)

    - [Example](./examples/question_answering.ipynb): See an example of how to run the code with your data to obtain results without the need to run the UI.

- [Risk Networks](./toolkit/risk_networks/README.md)

    - [Example](./examples/risk_networks/main.ipynb): See an example of how to run the code with your data to obtain results without the need to run the UI.

:construction: Code-only workflows in progress: 

- Data Synthesis
- Group Narratives
- Record Matching

## Running the UI (Streamlit) 

### Running via shell

- Poetry

    `poetry run poe run_streamlit`
- Pip

    `python -m streamlit run app/Home.py`

### Running with docker

Download and install docker: https://www.docker.com/products/docker-desktop/

Then, in the root folder, run:

`docker build . -t intelligence-toolkit`

After building, run the docker container with:

`docker run -d -p 8501:8501 intelligence-toolkit`

Open [localhost:8501](http://localhost:8501)

## Building a Windows executable

**Note: can only build on Windows**

We use [Pynsist](https://pynsist.readthedocs.io/en/latest/) together with [NSIS (Nullsoft Scriptable Install System)](https://nsis.sourceforge.io/) to build an executable for Windows. This packages the whole project and its dependencies (including Python) into an .exe, which when installed will run the Intelligence Toolkit on the user's localhost.

To build the .exe locally, you need to install pynsis with `pip install pynsist` and NSIS by [downloading it here](https://nsis.sourceforge.io/Main_Page).

Next, run `.\installer_script.ps1` in the root of the app to perform the following steps:
- download wkhtmltox from the source (needed to generate PDF reports). 
- build an .exe into build\nsis.

Once finished building, you can install the application by running the .exe and open the shortcut to launch intelligence-toolkit at http://localhost:8503 in your web browser.

## Deploying with Azure

In [this tutorial](https://dev.to/keneojiteli/deploy-a-docker-app-to-app-services-on-azure-5d3h), you can learn how to create the necessary services in azure.

From there, you can deploy it manually as described, or use [our YAML file](/.vsts-ci.yml) to automatically deploy to your environment. 

# Lifecycle Scripts

For Lifecycle scripts it utilizes [Poetry](https://python-poetry.org/docs#installation) and [poethepoet](https://pypi.org/project/poethepoet/) to manage build scripts.


Available scripts are:

- `poetry run poe test` - This will execute unit tests.
- `poetry run poe check` - This will perform a suite of static checks across the package, including:
  - formatting
  - documentation formatting
  - linting
  - security patterns
  - type-checking
- `poetry run poe fix` - This will apply any available auto-fixes to the package. Usually this is just formatting fixes.
- `poetry run poe fix_unsafe` - This will apply any available auto-fixes to the package, including those that may be unsafe.
- `poetry run poe format` - Explicitly run the formatter across the package.

