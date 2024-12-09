#!/bin/bash

location=$1
webAppName=$2
resourceGroup=$3
subscription=$4
tenant=$5

# Check if Terraform is installed
if ! command -v terraform &> /dev/null
then
    echo "Terraform not found. Installing..."
    wget  https://releases.hashicorp.com/terraform/1.10.0/terraform_1.10.0_linux_amd64.zip -O terraform.zip
    unzip terraform.zip
    mv terraform /usr/local/bin/
    rm terraform.zip
else
    echo "Terraform is already installed."
fi

# Verify Terraform version
terraform --version

# Download Terraform configuration files
echo "Downloading Terraform configuration files..."
wget https://raw.githubusercontent.com/microsoft/intelligence-toolkit/refs/heads/terraform/deploy/azure/main.tf -O ./main.tf
wget https://raw.githubusercontent.com/microsoft/intelligence-toolkit/refs/heads/terraform/deploy/azure/variables.tf -O ./variables.tf 
#Create modules/auth folder
mkdir modules
mkdir modules/auth
wget https://raw.githubusercontent.com/microsoft/intelligence-toolkit/refs/heads/terraform/deploy/azure/modules/auth/main.tf -O ./modules/auth/main.tf
wget https://raw.githubusercontent.com/microsoft/intelligence-toolkit/refs/heads/terraform/deploy/azure/modules/auth/variables.tf -O ./modules/auth/variables.tf 

# Initialize Terraform
terraform init

# Apply Terraform configuration
terraform apply -auto-approve -var "location=$location" -var "webAppName=$webAppName" -var "resourceGroup=$resourceGroup" -var "subscription=$subscription" -var "tenant=$tenant"
