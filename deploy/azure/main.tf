{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "webAppName": {
      "type": "string"
    },
    "location": {
      "type": "string"
    },
    "clientId": {
      "type": "string",
      "defaultValue": ""
    },
    "disableAuth": {
      "type": "bool",
      "defaultValue": false
    },
    "openaiApiKey": {
      "type": "string",
      "defaultValue": ""
    },
    "aiType": {
      "type": "string",
      "defaultValue": ""
    },
    "aiManagedIdentity": {
      "type": "bool",
      "defaultValue": false
    },
    "aiEndpoint": {
      "type": "string",
      "defaultValue": ""
    },
    "tags": {
      "type": "object",
      "defaultValue": {}
    },
    "alwaysOn": {
      "type": "bool",
      "defaultValue": true
    },
    "sku": {
      "type": "string",
      "defaultValue": "Premium v3"
    },
    "skuCode": {
      "type": "string",
      "defaultValue": "P0V3"
    },
    "workerSize": {
      "type": "string",
      "defaultValue": "8"
    },
    "workerSizeId": {
      "type": "string",
      "defaultValue": "8"
    },
    "numberOfWorkers": {
      "type": "string",
      "defaultValue": "1"
    }
  },
  "variables": {
    "hostingPlanName": "[concat(parameters('webAppName'), '-plan')]",
    "ftpsState": "FtpsOnly",
    "linuxFxVersion": "sitecontainers",
    "siteContainerName": "main",
    "redirectUrl": "[concat('https://', parameters('webAppName'), '.azurewebsites.net')]"
  },
  "resources": [
    {
      "apiVersion": "2019-10-01",
      "name": "pid-02837e98-dc6a-4353-9712-eb2e50086e2c-partnercenter",
      "type": "Microsoft.Resources/deployments",
      "properties": {
        "mode": "Incremental",
        "template": {
          "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
          "contentVersion": "1.0.0.0",
          "resources": []
        }
      }
    },
    {
      "apiVersion": "2022-03-01",
      "name": "[parameters('webAppName')]",
      "type": "Microsoft.Web/sites",
      "location": "[parameters('location')]",
      "tags": "[if(contains(parameters('tags').value, 'Microsoft.Web/sites'), parameters('tags').value['Microsoft.Web/sites'], json('null'))]",
      "dependsOn": [
        "[variables('hostingPlanName')]"
      ],
      "properties": {
        "name": "[parameters('webAppName')]",
        "siteAuthEnabled": true,
        "siteConfig": {
          "appSettings": [
            {
              "name": "WEBSITES_ENABLE_APP_SERVICE_STORAGE",
              "value": "false"
            },
            {
              "name": "DOCKER_ENABLE_CI",
              "value": "true"
            },
            {
              "name": "OPENAI_API_KEY",
              "value": "[parameters('openaiApiKey')]"
            },
            {
              "name": "OPENAI_TYPE",
              "value": "[parameters('aiType')]"
            },
            {
              "name": "AZURE_AUTH_TYPE",
              "value": "[if(equals(parameters('aiManagedIdentity'), 'true'), 'Managed Identity', 'Azure Key')]"
            },
            {
              "name": "AZURE_OPENAI_ENDPOINT",
              "value": "[parameters('aiEndpoint')]"
            }
          ],
          "linuxFxVersion": "[variables('linuxFxVersion')]",
          "alwaysOn": "[parameters('alwaysOn')]",
          "ftpsState": "[variables('ftpsState')]"
        },
        "serverFarmId": "[concat('/subscriptions/', subscription().subscriptionId,'/resourcegroups/', resourceGroup().id, '/providers/Microsoft.Web/serverfarms/', variables('hostingPlanName'))]",
        "clientAffinityEnabled": false,
        "httpsOnly": true,
        "publicNetworkAccess": "Enabled"
      },
      "resources": [
        {
          "type": "Microsoft.Web/sites/basicPublishingCredentialsPolicies",
          "apiVersion": "2022-09-01",
          "name": "[concat(parameters('webAppName'), '/scm')]",
          "properties": {
            "allow": false
          },
          "dependsOn": [
            "[resourceId('Microsoft.Web/Sites', parameters('webAppName'))]"
          ]
        },
        {
          "type": "Microsoft.Web/sites/basicPublishingCredentialsPolicies",
          "apiVersion": "2022-09-01",
          "name": "[concat(parameters('webAppName'), '/ftp')]",
          "properties": {
            "allow": false
          },
          "dependsOn": [
            "[resourceId('Microsoft.Web/Sites', parameters('webAppName'))]"
          ]
        }
      ]
    },
    {
      "type": "Microsoft.Web/sites/config",
      "apiVersion": "2022-09-01",
      "name": "[concat(parameters('webAppName'), '/authsettingsV2')]",
      "dependsOn": [
        "[resourceId('Microsoft.Web/Sites', parameters('webAppName'))]"
      ],
      "properties": {
        "globalValidation": {
          "requireAuthentication": "[equals(parameters('disableAuth'), 'false')]",
          "unauthenticatedClientAction": "RedirectToLoginPage"
        },
        "identityProviders": {
          "azureActiveDirectory": {
            "enabled": "[equals(parameters('disableAuth'), 'false')]",
            "registration": {
              "clientId": "[parameters('clientId')]",
              "openIdIssuer": "[concat('https://login.microsoftonline.com/',subscription().tenantId,'/v2.0')]"
            },
            "validation": {
              "allowedAudiences": [ "[parameters('clientId')]" ]
            }
          }
        }
      }
    },
    {
      "type": "Microsoft.Web/sites/sitecontainers",
      "apiVersion": "2023-12-01",
      "name": "[format('{0}/{1}', parameters('webAppName'), variables('siteContainerName'))]",
      "properties": {
        "image": "ghcr.io/microsoft/intelligence-toolkit:latest",
        "targetPort": "80",
        "isMain": true,
        "startUpCommand": "",
        "authType": "Anonymous"
      },
      "dependsOn": [
        "[resourceId('Microsoft.Web/sites', parameters('webAppName'))]"
      ]
    },
    {
      "apiVersion": "2023-12-01",
      "name": "[variables('hostingPlanName')]",
      "type": "Microsoft.Web/serverfarms",
      "location": "[parameters('location')]",
      "kind": "linux",
      "dependsOn": [],
      "properties": {
        "name": "[variables('hostingPlanName')]",
        "workerSize": "[parameters('workerSize')]",
        "workerSizeId": "[parameters('workerSizeId')]",
        "numberOfWorkers": "[parameters('numberOfWorkers')]",
        "reserved": true,
        "zoneRedundant": false
      },
      "sku": {
        "Tier": "[parameters('sku')]",
        "Name": "[parameters('skuCode')]"
      }
    }
  ],
  "outputs": {
    "webAppName": {
      "type": "string",
      "value": "[parameters('webAppName')]"
    },
    "hostingPlanName": {
      "type": "string",
      "value": "[variables('hostingPlanName')]"
    },
    "location": {
      "type": "string",
      "value": "[parameters('location')]"
    },
    "tags": {
      "type": "object",
      "value": "[parameters('tags')]"
    }
  }
}