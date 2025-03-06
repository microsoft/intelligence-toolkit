# Deploying the App Using Azure Marketplace
This guide will help you deploy your application using Azure Marketplace, even if you're not a technical expert.

### Prerequisites
Before you start, make sure you have the following:

1. **Azure Account**: At minimum `Contributor` role assignment to deploy the resources in your Azure subscription.
2. **Azure Subscription**: An active Azure subscription.
3. **OpenAI or Azure OpenAI instance**: This will provide the AI capabilities required by the application. If you don't have one, you can find [instructions here](../../OPENAI.md)

### Steps to Deploy the App

It is recommended that you use Entra ID for authentication.

#### Creating an app registration for website authentication

- Open your web browser and go to [Microsoft Entra admin center](https://entra.microsoft.com/#view/Microsoft_AAD_RegisteredApps/ApplicationsListBlade)
- On the left panel click on `Applications` then `App registrations`
- Click on `+ New registration`.
    - **Give it an identifiable name**
    - **Supported account types**:
        - Accounts in this organizational directory only ([Your Organization] only - Single tenant) 
- **Redirect URI**:
    - Select `Web` as platform and in the URL insert the following URL, with `[webAppName]` being the name you'll give to the app on the next steps: 
        - `https://[webAppName].azurewebsites.net/.auth/login/aad/callback`
    - Click on `Register`
    - Copy the value of `Application (client) ID` to be used when creating the app in the next steps.

#### Deploy
See details on pricing [here](https://azure.microsoft.com/en-us/pricing/details/app-service/linux/) (We default to `Premium v3 P0v3` Plan)

1. **Go to Azure Marketplace**:

    - In your browser, open [Intelligence Tooklit on Marketplace here](https://portal.azure.com/#create/msr-resilience.itk_app_servicewebapp)

2. **Select the Application**:
    - Click on the application from the search results to open its details page.

3. **Click "Create"**:
    - Click the "Create" button with Plan `Web App` selected to start the deployment process.

4. **Configure Deployment Settings**:
    - You will be redirected to the [Azure portal](https://portal.azure.com) to set up the deployment.
    - Fill in the required information:
        - **Subscription**: Choose your Azure subscription.
        - **Resource Group**: Select an existing group or create a new one.
        - **Region**: Choose the region where you want to deploy the app (the closer to y ou, the better).
        - **Web app name**: Enter a unique name for your web app.
            - This will create the URl you'll access:
                `webappname`.azurewebsites.net
        - **Disable authentication**:
            - Having authentication enabled is **strongly** recommended. But if for any case you want to test the app making sure that no private data will be used and you're ok with the website being open to the internet to access, check this option.
        - **App registration client ID**:
            - In here you'll insert the `Application (client) ID` value you created earlier.
        - **Agent VM size**
            - Pre-set options on the computation size of your app. Prices may vary. The default optin is the cheaper one. Higher values are better for more users, and/or if you want your app to be faster.
                - [Click here](https://azure.microsoft.com/en-us/pricing/details/app-service/linux/) for more information on pricing.
        - **AI settings**:
            - This will configure how the app will access an AI instance.
            - **AI type**: 
                - OpenAI or Azure OpenAI
            - **Use Managed Identity**:
                - Check this if type is Azure OpenAI and you don't have a key, but the user accessing the app has a role assigment in the Azure OpenAI resource.
            - **Endpoint**:
                - If Azure OpenAI, insert the endpoint for it.
            - **Key**:
                - Only if Managed Identity is not checked, insert your OpenAI or Azure OpenAI key here.
        - **Tags** (Optional): Add any tags you want to use to organize your resources.

5. **Review and Create**:
    - Review the settings you entered to make sure everything is correct.
    - Click the "Review + create" button to validate the settings.
    - Once validation is complete, click the "Create" button to start the deployment.

6. **Monitor Deployment**:
    - The deployment process will begin, and you can watch its progress in the Azure portal.
    - When the deployment is finished, you will get a notification.

7. **Access the Deployed Application**:
    - Go to the resource group where the application was deployed.
    - Find the web application resource and click on it to open its details page.
    - It takes a few minutes for the application to be ready. Wait about 5-10 and then use the provided URL (`webappname`.azurewebsites.net) to access your deployed application.

By following these steps, you can successfully deploy your application using Azure Marketplace.