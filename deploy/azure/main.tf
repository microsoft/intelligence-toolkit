provider "azurerm" {
  subscription_id = var.subscription_id
  features {}
}

resource "azurerm_resource_group" "az_rg" {
  name     = var.az_rg_name
  location = var.location
}

resource "azurerm_service_plan" "az_asp" {
  os_type="Linux"
  name                = concat(var.az_webapp_name, "-plan")
  location            = azurerm_resource_group.az_rg.location
  resource_group_name = azurerm_resource_group.az_rg.name

  sku_name = "S1"

  depends_on = [
    azurerm_resource_group.az_rg
  ]
}

module "auth" { 
  source = "./modules/auth" 
  
  enable_auth = var.enable_auth
  application_display_name = concat(var.az_webapp_name, "-auth")
  tenant_id = var.tenant_id 
  redirect_uris = ["https://${var.az_webapp_name}.azurewebsites.net/.auth/login/aad/callback"]
}

resource "azurerm_linux_web_app" "az_webapp" {
  name                = var.az_webapp_name
  location            = azurerm_resource_group.az_rg.location
  resource_group_name = azurerm_resource_group.az_rg.name
  service_plan_id     = azurerm_service_plan.az_asp.id
  https_only          = true

  depends_on = [
    azurerm_service_plan.az_asp
  ]

  site_config {
    application_stack {
      docker_image_name = "${var.ghcr_image}"
      docker_registry_url = "https://ghcr.io"

    }
  }

  dynamic "auth_settings_v2" {
    for_each = var.enable_auth ? [1] : []

    content {
      auth_enabled           = true
      unauthenticated_action = "Return401"
      require_authentication = true
      require_https          = true

      active_directory_v2 {
        client_id              = module.auth.application_id
        tenant_auth_endpoint   = "https://sts.windows.net/${var.tenant_id}/v2.0"
        allowed_audiences      = ["api://example"]
      }

      login {
        token_store_enabled = true
      }
    }
  }
}