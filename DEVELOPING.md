# Developing 

## Requirements

- Python 3.10 ([Download](https://www.python.org/downloads/))

## Set up:
1. Set up virtual environment:

    `python -m venv ./venv`

    or
    
    `python3 -m venv ./venv`

2. Install wkhtmltopdf to be able to generate the final story in PDF:

    Windows: [Download wkhtmltopdf installer](https://wkhtmltopdf.org/downloads.html)

    Linux:  `sudo apt-get install wkhtmltopdf`

    MacOS: `brew install homebrew/cask/wkhtmltopdf`

## OpenAI Key

### Linux:
Open /venv/bin/activate, add the following lines at the end of the file:
```
    # set environment variables
    export OPENAI_API_KEY=<OPENAI_API_KEY>

    # if Azure OpenAI, include the following information too:
    export OPENAI_TYPE="Azure OpenAI"
    export AZURE_OPENAI_VERSION=2023-12-01-preview
    export AZURE_OPENAI_ENDPOINT="https://<ENDPOINT>.azure.com/"
```

### Windows:
Open venv/Scripts/Activate.ps1, add the following lines after line 167:
```
    $env:OPENAI_API_KEY="<OPENAI_API_KEY>"

    # if Azure OpenAI, include the following information too:

    $env:OPENAI_TYPE="Azure OpenAI"
    $env:AZURE_OPENAI_VERSION="2023-12-01-preview"
    $env:AZURE_OPENAI_ENDPOINT="https://<ENDPOINT>.openai.azure.com/"
``` 

### Running

1. Run the activate: 

    `source venv/bin/activate`  (Linux)

    `.\venv\Scripts\Activate` (Windows with Powershell)

2. Install all the dependencies with pip:

    `pip install -r requirements.txt`

3. Run the project using streamlit: 

    
    `streamlit run app/Home.py`


## Running with docker

Download and install docker: https://www.docker.com/products/docker-desktop/

Then, in the root folder, run:

`docker build . -t intel-toolkit`

After building, run the docker container with:

`docker run -d -p 8501:8501 intel-toolkit`

Open [localhost:8501](http://localhost:8501)

## Building a Windows executable

We use [Pynsist](https://pynsist.readthedocs.io/en/latest/), that with [NSIS (Nullsoft Scriptable Install System)](https://nsis.sourceforge.io/) builds an executable for Windows, which packages the whole project and what it needs to run (including Python) into an .exe, that when installed will run the project on the user's localhost.

For you to build locally, you will need to have pynsis intalled with `pip install pynsist` and install NSIS [downloading it here](https://nsis.sourceforge.io/Main_Page).

**Tip**: Use Windows to build it, not Linux.

Run `.\installer_script.ps1` in the root of the app.
It will download wkhtmltox from the source, that's needed to generate reports. 
Then it will download python-louvain wheel, because it's not on pypi and it's needed for pynsist.
Then it will build an .exe into build\nsis.

It takes a while to finish, but then you can install it and open the shortcut to open intelligence-toolkit into http://localhost:8501

# Deploying

- In [this tutorial](https://dev.to/keneojiteli/deploy-a-docker-app-to-app-services-on-azure-5d3h), you can check how to create the services in azure.
    - From there, you can deploy it manually like it's written, or use [our YAML file](/.vsts-ci.yml) to automatically deploy to your environment if you configure it. 