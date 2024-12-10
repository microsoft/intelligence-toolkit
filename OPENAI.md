# OpenAI or Azure OpenAI Instance
Before you can deploy the application, you need to have an OpenAI or Azure OpenAI instance. This instance will provide the AI capabilities required by the application. Below are the steps to create an OpenAI or Azure OpenAI instance.

## Creating an OpenAI Instance
See pricing details [here](https://openai.com/api/pricing/)
1. **Sign Up for OpenAI**:
    - Go to the [OpenAI website](https://platform.openai.com/login/).
    - Login or sign in (you can use your ChatGPT account)
        - You will need a phone number to confirm your account.

2. **Create the project**
    - Click on Create a new project.
    - Give it an identifiable name and click create.

3. **Add billing details**
    - On the top bar on the right, click on your profile.
    - Click on `Billing` on the left panel.
    - Add your payment details.

4. **Get API Key**:
    - On the top bar on the right, click on your profile.
    - Click on `API keys` on the left panel.
    - Click on `+ Create new secret key` on the top right.
    - Give it an identifiable name and select the project you created.
    - Click `Create secret key`.
    - Copy your key and store it safely. It won't show again and if lost you'll need to create a new one.
    - Use these to configure your access when deploying intelligence-toolkit app or using the `Settings` page.




## Creating an Azure OpenAI Instance
See pricing details [here](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)

1. **Log in to the Azure portal**:
    - Go to the [Azure portal](https://portal.azure.com).
        - If you don't have an account, click on "Create one" and follow the steps to set up your account.

2. **Create the resource**:
    - Click on "Create a resource" and search for "Azure OpenAI".
    - Select "Azure OpenAI Service" and click "Create".
    - Fill in the required details (subscription, resource group, region, etc.).
    - Click next until the Review + submit step.
    - If validation passes, click create.

3. **Deploy the models**:
    - Go to the resource page.
    - Click on `Go to Azure AI Foundry portal`
    - Click on `Deployments` on the left panel.
    - Create the AI model
        - Click on `Deploy model` and `Deploy base model`.
        - Choose `gpt-4o` then `Confirm` and `Deploy`
    - Create the embedding model
        - Click on `Deploy model` and `Deploy base model`.
        - Choose `text-embedding-3-small` then `Confirm` and `Deploy`

4. **Get the Azure OpenAI key**:
    - In the resource page on Azure portal, go to "Keys and Endpoint" section.
    - Copy  one of the keys (KEY 1 or KEY 2) and Endpoint URL.
    - Use these to configure your access when deploying intelligence-toolkit app or using the `Settings` page.