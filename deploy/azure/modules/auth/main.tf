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
resource "azuread_application" "app_registration" {
  count = var.enable_auth ? 1 : 0
  display_name = var.application_display_name

  web { 
    redirect_uris = var.redirect_uris 
  }
}

# Create a service principal
resource "azuread_service_principal" "app_service_principal" {
  count = var.enable_auth ? 1 : 0
  client_id = var.enable_auth ? azuread_application.app_registration[count.index].client_id : null
}

output "application_id" {
  description = "The Application ID of the Azure AD application"
  value       = var.enable_auth ? azuread_application.app_registration[0].client_id : null
}