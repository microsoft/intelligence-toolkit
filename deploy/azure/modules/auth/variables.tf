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

variable "enable_auth" {
  type = bool
  default = false
}