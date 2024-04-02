# Intelligence Toolkit
The Intelligence Toolkit is a suite of interactive workflows for creating AI intelligence reports from real-world data sources. The toolkit is designed to help users identify patterns, answers, relationships, and risks within complex datasets, with generative AI ([OpenAI GPT models](https://platform.openai.com/docs/models/)) used to create reports on findings of interest.


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
```

### Windows:
Open venv/Scripts/Activate.ps1, add the following lines after line 167:
```
    $env:OPENAI_API_KEY="<OPENAI_API_KEY>"
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


# Deploying

- In [this tutorial](https://dev.to/keneojiteli/deploy-a-docker-app-to-app-services-on-azure-5d3h), you can check how to create the services in azure.
    - From there, you can deploy it manually like it's written, or use [our YAML file](/.vsts-ci.yml) to automatically deploy to your environment if you configure it. 

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
