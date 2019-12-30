---
layout: post
comments: true
title: How to create premium azure block blob
---

The premium block blob account is not ready in azure portal, but could be
created by Azure CLI. 

1. Install the [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)
2. Login 
```shell
az login
```
3. Set the subscription as active
```shell
az account set --subscription "Microsoft Azure Internal - VIG"
```
4. Create the account
```shell
az storage account create --location "West US" --name vig --resource-group VIGResourceGroup --kind "BlockBlobStorage" --sku "Premium_LRS"
```
