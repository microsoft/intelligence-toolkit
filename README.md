# Introduction 

pip install -r requirements.txt

streamlit run app/Home.py

There's also a [windows executable file](./build/nsis/) that you could install with your machine, and you will have intelligence-toolkit out of the box.

## PDF export
Install wkhtmltopdf to be able to generate the final story in PDF:

Windows: [Download wkhtmltopdf installer](https://wkhtmltopdf.org/downloads.html)

Linux:  `sudo apt-get install wkhtmltopdf`

MacOS: `brew install homebrew/cask/wkhtmltopdf`

# Building a Windows executable

We use [Pynsist](https://pynsist.readthedocs.io/en/latest/), that with [NSIS (Nullsoft Scriptable Install System)](https://nsis.sourceforge.io/) builds an executable for Windows, which packages the whole project and what it needs to run (including Python) into an .exe, that when installed will run the project on the user's localhost.

For you to build locally, you will need to have pynsis intalled with `pip install pynsist` and install NSIS [downloading it here](https://nsis.sourceforge.io/Main_Page).

**Tip**: Had problems when trying to run on Linux, so it's easier to develop in Windows.

Run `pynsist .\installer.cfg` in the root of the app, and it will build an .exe into build\nsis.

This install the app, then through Windows menu find the app and open it. It will start streamlit server and open a page in localhost. It might take a while to the app to load, so don't worry about 'page not found' warning.


# Project

> This repo has been populated by an initial template to help get you started. Please
> make sure to update the content to build a great experience for community-building.

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

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
