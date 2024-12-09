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
  name                = "${var.az_webapp_name}-plan"
  location            = azurerm_resource_group.az_rg.location
  resource_group_name = azurerm_resource_group.az_rg.name

  sku_name = "S1"

  depends_on = [
    azurerm_resource_group.az_rg
  ]
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

}