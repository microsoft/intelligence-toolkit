# Deploying the app

Deploying your app effectively is crucial for making it accessible to users. We present three robust options that cater to different requirements and preferences, whether you're looking for simplicity, scalability, or specific infrastructure.

- **Using Azure:** A powerful option for those who are already integrated into the Microsoft ecosystem or need advanced cloud services. Azure offers robust performance, scalability, and a suite of tools that are beneficial for apps requiring Microsoft-specific integrations or high availability.

- **Using AWS:** Known for its versatility and comprehensive cloud solutions, AWS is suitable for developers seeking a highly scalable and flexible deployment environment. It is apt for those with diverse app requirements and offers a range of services, from simple hosting to complex machine learning models.

Each of these options has its own strengths, and your choice will depend on the specific needs and constraints of your project. The following sections provide detailed guides on deploying your app using each of these platforms.

# Requirements for the app

To deploy your app, you will need:

1. An active OpenAI account ([create here](https://platform.openai.com/login)).
2. An OpenAI API key ([create here](https://platform.openai.com/account/api-keys)).

# Azure

#### Recommended configuration:

- *Minimum disk space*: 8GB 

- *Minimum memory*: 4GB
    - If too many users using at the same time, it might need to be higher.

## Azure

You can modify the code and deploy the container, or use our default container hosted on ghcr.io.

### Deploying your container
#### TODO

### Using ghcr.io

**Prerequisites**

- Azure Account: Ensure you have an [active Azure subscription](https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account?msockid=1e4bc940d7cf6738158eda91d616667e).

- Terraform: Install Terraform on your local machine. You can download it from [terraform.io](https://developer.hashicorp.com/terraform/install?product_intent=terraform).

- Azure CLI: Install Azure CLI. You can download it from [docs.microsoft.com](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

**Steps**

1. Set Up Azure CLI
    Login to Azure:

    `az login` 

    This will open a browser window for you to authenticate.
    Set the Subscription:
    If you have multiple subscriptions, set the one you wish to use:

    `az account set --subscription "your-subscription-id"`

    Create a Directory for Your Project:

    ```
    mkdir my-terraform-app  
    cd my-terraform-app  
    ```
 
2. Use our Terraform Configuration File:
    Download the [terraform configuration file here](https://github.com/microsoft/intelligence-toolkit/blob/main/deploy/azure/main.tf)


    -Modify the `variables` default field to match your desired resource configuration

        - az_webapp_name (change necessary. Should be unique within Global Azure)

        - az_rg_name (change optional)

        - location (change optional)

        - az_asp_name (change optional)

3. Initialize Terraform


    `terraform init`
    This command downloads the Azure provider and sets up your workspace.

4. Create an Execution Plan
 
    Plan the Deployment:

    `terraform plan` 
    This command creates an execution plan, which lets you preview the changes that Terraform will make to your infrastructure.

5. Apply the Execution Plan

    Deploy the Resources:

    `terraform apply` 
    Terraform will prompt you for confirmation before making any changes. Type yes and press Enter.

6. Verify Deployment

    Check the Resources in Azure Portal:
    Go to the Azure Portal and verify that the resources have been created.

    Check the deployed URL:

`<az_webapp_name>.azurewebsites.net`

# AWS

Wait for step 1 to be set as complete before starting step 2. The whole process will take up to 20 minutes.

1. Launch the infrastructure deploy:

    - Give it a sugestive name since you'll be using it in the next step.

    [![launch-stack](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=itk-infra-stack&templateURL=https://s3.us-east-1.amazonaws.com/cf-templates-19n482mly1fba-us-east-1/2024-10-07T124926.165Z3xc-infrastructure.yaml)

2. Launch the code deploy
    - In VPC Configuration, you should select the resources created by the previous step: <u>VPCId, PublicSubnetAId, PublicSubnetBId, PrivateSubnetAId, PrivateSubnetBId</u>

    [![launch-stack](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=itk-code-stack&templateURL=https://s3.us-east-1.amazonaws.com/cf-templates-19n482mly1fba-us-east-1/2024-10-07T125858.730Zlsu-2-development.yaml)


Once step 2 it's complete, in the output tab, you'll see the deployed URL.

## Environment configuration

To effectively manage user access and configuration settings in your app's deployment environment, here are some key parameters and steps for setting up your environment configuration. This guide covers how to handle these settings in Streamlit Cloud, Azure, and AWS deployments.

## Environment Configuration

- **Hiding the Settings Page**

  Use `HIDE_SETTINGS=TRUE` to conceal the Settings page. This prevents users from altering configurations that could impact the experience for other users.

- **OpenAI API Key**

  Set `OPENAI_API_KEY="your-key"` to ensure secure access to OpenAI services. This key allows interaction with OpenAI's API while keeping it confidential.

- **Authentication**

  Enable `AUTH_ENABLED=TRUE` to restrict app access to authorized users with credentials defined in a `.secrets.toml` file under the `.streamlit` directory:

  ```toml
  [passwords]
  user_test = "user123"
  ...
  ```

### Inserting the `secrets.toml` file into Web App Deployments (Azure or AWS)

When deploying your app on Azure or AWS, you may need to configure user credentials as environment variables. Here's how you can accomplish this:

- **User Credentials Environment Variable**

  Add credentials in the format `user:password`, separated by semicolons (`;`), to your web app's environment variables:

  ```plaintext
  USER_CREDENTIALS="user1:pwd1;user2:pwd2"
  ```

This setup ensures secure handling of user authentication and sensitive configurations across different deployment platforms. By using these configurations, you can maintain control over user access and protect essential settings.



