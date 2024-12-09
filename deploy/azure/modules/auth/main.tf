# Configure Terraform
terraform {
  required_providers {
    azuread = {
      source  = "hashicorp/azuread"
      version = "3.0"
    }
  }
}

# Configure the Azure Active Directory Provider
provider "azuread" {
  tenant_id = var.tenant_id
}

# Retrieve domain information
data "azuread_domains" "domains" {
  only_initial = true
}

# Create an application
resource "azuread_application_registration" "app_registration" {
  display_name = var.application_display_name
}

resource "azuread_application" "app_registration_config" {
  display_name = var.application_display_name

  web { 
    redirect_uris = var.redirect_uris 
  }
}

# Create a service principal
resource "azuread_service_principal" "app_service_principal" {
  client_id = azuread_application_registration.app_registration.client_id
}

output "application_id" {
  description = "The Application ID of the Azure AD application"
  value       = azuread_application_registration.app_registration.client_id
}