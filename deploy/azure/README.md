# Deploying the App Using Azure Marketplace
This guide will help you deploy your application using Azure Marketplace, even if you're not a technical expert.

### Prerequisites
Before you start, make sure you have the following:

1. **Azure Subscription**: An active Azure subscription.
1. **OpenAI or Azure OpenAI instance**: This will provide the AI capabilities required by the application. If you don't have one, you can find [instructions here](../../OPENAI.md)
2. **Azure Account**: At minimum `Contributor` role assignment to deploy the resources in your Azure subscription.

### Steps to Deploy the App
1. **Go to Azure Marketplace**:
    - Open your web browser and go to Azure portal, search in the top bar for `Marketplace`.
    - If you're not logged in, you may need to log in to your Azure account. 
    - Use the search bar to find the `Intelligence Toolkit` application.

2. **Select the Application**:
    - Click on the application from the search results to open its details page.

3. **Click "Create"**:
    - Click the "Create" button with Plan `Web App` selected to start the deployment process.

4. **Configure Deployment Settings**:
    - You will be redirected to the Azure portal to set up the deployment.
    - Fill in the required information:
        - **Subscription**: Choose your Azure subscription.
        - **Resource Group**: Select an existing group or create a new one.
        - **Region**: Choose the region where you want to deploy the app (the closer to y ou, the better).
        - **Web App Name**: Enter a unique name for your web app.
            - This will create the URl you'll access:
                `webappname`.azurewebsites.net
        - **Service Principal Type**:
            - To create a new authentication app, leave as `Create New`
            - Click change Selection to change its name or leave it as the default.
        - **AI Settings**:
            - This will configure how the app will access an AI instance.
            - **AI Type**: 
                - OpenAI or Azure OpenAI
            - **Use Managed Identity**:
                - Check this if type is Azure OpenAI and you don't have a key, but the user accessing the app have permission to it.
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