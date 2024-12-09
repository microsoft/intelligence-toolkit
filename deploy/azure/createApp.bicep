param appDisplayName string
param subscriptionId string

resource userAssignedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2018-11-30' = {
  name: '${appDisplayName}-identity'
  location: resourceGroup().location
}

resource deploymentScript 'Microsoft.Resources/deploymentScripts@2023-08-01' = {
  name: 'createAppRegistration'
  location: resourceGroup().location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      userAssignedIdentity.id: {}
    }
  }
  kind: 'AzureCLI'
  properties: {
    azCliVersion: '2.49.0'
    scriptContent: 'az account set --subscription ${subscriptionId}; az ad app create --display-name ${appDisplayName}'
    timeout: 'PT30M'
    cleanupPreference: 'OnSuccess'
    retentionInterval: 'P1D'
  }
}
