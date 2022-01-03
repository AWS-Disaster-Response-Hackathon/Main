# AWS-Disaster-Response-Hackathon Rain fall Predictor --> Flood Predictor

This Machine Learning project is created for AWS-Disaster-Response-Hackathon which uses 2 phase prediction using Machine Learning algorithm.

**Phase-1 :** Based on all historical monthy rain fall data using Nonlinear Autoregressive Exogenous (NARX) neural network model to predict rainfall for next Months , Weeks , Days or hours based on provided data for training.
For experiment purpose i used monthly average rain fall data focusing on specific region.

**Phase-2 :** Once rain fall model predicts the rain fall that data feeds into another Machine Learning model which is built using Random Forest Classification Model. This model predicts the chances of flood due to this overall rain fall.
Predictions are done considering +-10% error , so model output in following way with overall % confidence level of the alert.

 ***************Flood Prediction Confusion Matrix Based on Rain***************
[[10  0]
 [ 1  9]]
 
 ***************Send Alert for flood***************
Red Zone : There are 100% chance the region will be hit by flood during month of 7/2017
Amber Zone : There are 66.67% chance the region will be hit by flood during month of 8/2017 
