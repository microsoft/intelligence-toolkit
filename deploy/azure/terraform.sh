#!/bin/bash

# Check if Terraform is installed
if ! command -v terraform &> /dev/null
then
    echo "Terraform not found. Installing..."
    curl -sLo terraform.zip https://releases.hashicorp.com/terraform/1.10.0/terraform_1.10.0_linux_amd64.zip
    unzip terraform.zip
    sudo mv terraform /usr/local/bin/
    rm terraform.zip
else
    echo "Terraform is already installed."
fi

# Verify Terraform version
terraform --version

# Download Terraform configuration files
echo "Downloading Terraform configuration files..."
curl -o ./main.tf https://raw.githubusercontent.com/microsoft/intelligence-toolkit/refs/heads/terraform/deploy/azure/main.tf
curl -o ./variables.tf https://raw.githubusercontent.com/microsoft/intelligence-toolkit/refs/heads/terraform/deploy/azure/variables.tf
curl -o ./modules/auth/main.tf https://raw.githubusercontent.com/microsoft/intelligence-toolkit/refs/heads/terraform/deploy/azure/modules/auth/main.tf
curl -o ./modules/auth/variables.tf https://raw.githubusercontent.com/microsoft/intelligence-toolkit/refs/heads/terraform/deploy/azure/modules/auth/variables.tf

# Initialize Terraform
terraform init

# Apply Terraform configuration
terraform apply -auto-approve
