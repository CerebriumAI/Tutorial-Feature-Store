# Real-time Credit Scoring with Feast on AWS

## Overview

![credit-score-architecture@2x](https://user-images.githubusercontent.com/6728866/132927464-5c9e9e05-538c-48c5-bc16-94a6d9d7e57b.jpg)

This tutorial demonstrates the use of Feast as part of a real-time credit scoring application.
* The primary training dataset is a loan table. This table contains historic loan data with accompanying features. The dataset also contains a target variable, namely whether a user has defaulted on their loan.
* Feast is used during training to enrich the loan table with zipcode and credit history features from a S3 files. The S3 files are queried through Redshift.
* Feast is also used to serve the latest zipcode and credit history features for online credit scoring using DynamoDB.

## Requirements

* Terraform (v1.0 or later)
* AWS CLI (v2.2 or later)

## Get Started

In order to get started with this tutorial you can open the Feature_Store.ipynb file and follow the steps.

Happy Coding!
