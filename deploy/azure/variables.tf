variable "enable_auth" {
  description = "Enable authentication" 
  type = bool 
  default = false 
}

variable "tenant_id" {
  description = "The Azure Active Directory tenant ID"
  type        = string
  default = ""
}

variable "application_display_name" {
  description = "The display name for the Azure AD application"
  type        = string
  default = "itk_app_reg"
}
variable "redirect_uris" {
    description = "List of redirect URIs for the application" 
    type = list(string) 
    default = []
}

variable "az_webapp_name" {
  type    = string
  default = ""
}

variable "subscription_id" {
  type    = string
  default = ""
}

variable "az_rg_name" {
  type    = string
  default = "webapp-itk-rg"
}

variable "location" {
  type    = string
  default = "East US"
}

variable "az_asp_name" {
  type    = string
  default = "webapp-itk-appserviceplan"
}

variable "ghcr_image" {
  type    = string
  default = "microsoft/intelligence-toolkit:latest"
}
