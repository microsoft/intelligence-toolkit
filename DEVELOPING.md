# Developing 

## Requirements

- Python 3.11 ([Download](https://www.python.org/downloads/))
- poetry ([Download](https://python-poetry.org/docs/#installing-with-the-official-installer))
- wkhtmltopdf (used to generate PDF reports)

    - Windows: ([Download](https://wkhtmltopdf.org/downloads.html))

    - Linux:  `sudo apt-get install wkhtmltopdf`

    - MacOS: `brew install homebrew/cask/wkhtmltopdf`


## Install Dependencies

`poetry install`

### LLM API access

#### Default values: 
```
OPENAI_API_MODEL="gpt-4o"
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

See the documentation and an example of how to run the code with your data to obtain results without the need to run the UI.
- [Anonymize Case Data](./app/anonymize_case_data/README.md)

    - [Example](./example_notebooks/anonymize_case_data.ipynb)

- [Compare Case Groups](./app/compare_case_groups/README.md)

    - [Example](./example_notebooks/compare_case_groups.ipynb)

- [Detect Case Patterns](./app/detect_case_patterns/README.md)

    - [Example](./example_notebooks/detect_case_patterns.ipynb)

- [Detect Entity Networks](./app/detect_entity_networks/README.md)

    - [Example](./example_notebooks/detect_entity_networks/main.ipynb)

- [Extract Record Data](./app/extract_record_data/README.md)

    - [Example](./example_notebooks/extract_record_data/main.ipynb)

- [Generate Mock Data](./app/generate_mock_data/README.md)

    - [Example](./example_notebooks/generate_mock_data/main.ipynb)

- [Match Entity Records](./app/match_entity_records/README.md)

    - [Example](./example_notebooks/match_entity_records/main.ipynb)
    
- [Query Text Data](./app/query_text_data/README.md)

    - [Example](./example_notebooks/query_text_data.ipynb)

## Running the UI (Streamlit) 

### Running via shell

`poetry run poe run_streamlit`

### Running with docker

Download and install docker: https://www.docker.com/products/docker-desktop/

Then, via shell, in the root folder, run:

`docker build . -t intelligence-toolkit`

After building, run the docker container:

- via shell:

    `docker run  -d -p 80:80 intelligence-toolkit --name intelligence-toolkit`

- via Docker GUI:


Open [localhost:80](http://localhost:80)

#### Do you want to share it? No need to build again.

Run

`docker save --output="intelligence_toolkit.tar" intelligence-toolkit`

then share the .tar file as you'd like.

In the target environment, in the same folder as the .tar file, run:

`docker load --input intelligence_toolkit.tar`

Once done, use the same command as above to run it.

## Deploying with AWS
Wait for step 1 to be set as complete before starting step 2. The whole process will take up to 20 minutes.

1. Launch the infrastructure deploy:

    - Give it a sugestive name since you'll be using it in the next step.

    [![launch-stack](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=itk-infra-stack&templateURL=https://s3.us-east-1.amazonaws.com/cf-templates-19n482mly1fba-us-east-1/2024-10-07T124926.165Z3xc-infrastructure.yaml)

2. Launch the code deploy
    - In VPC Configuration, you should select the resources created by the previous step: <u>VPCId, PublicSubnetAId, PublicSubnetBId, PrivateSubnetAId, PrivateSubnetBId</u>

    [![launch-stack](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=itk-code-stack&templateURL=https://s3.us-east-1.amazonaws.com/cf-templates-19n482mly1fba-us-east-1/2024-10-07T125858.730Zlsu-2-development.yaml)


Once step 2 it's complete, in the output tab, you'll see the deployed URL.

**Note: This code doesn't have auth, so this URL will be open to the internet.**

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

