################################################
#Predicting end accuracy using linear regression 
#This program performs 10-fold cross validation 
#with parameter optimization using the regression
#method of choice.

rm(list=ls())
library(caret)

data = read.csv("/Users/Charm/OneDrive/Insight_Work_Jan2016/Project/Code_from_CT/CT_AuditoryCommand_Level2.csv")
x= c(1,30:38, 41:43) 
data = data[,-x] #remove these columns
data_new = data[,-nearZeroVar(data)] #remove columns which have few data
ctrl = trainControl(method = "repeatedcv",repeats = 1) # Define training method
fitlm = train(EndAccuracy ~ ., # Select training feature
              data = data_new, # Select database
              method = "lm", # Select training method 
              preProcess = c("center","scale"),
              trControl = ctrl,
              tuneLength = 6,
              metric = "Rsquared") # Define pre-process parameters
summary(fitlm)
imp = varImp(fitlm) #feature importance
plot(imp)
imp$importance


################################################
#Predicting end accuracy using SVM
rm(list=ls())
library(caret)

data = read.csv("/Users/Charm/OneDrive/Insight_Work_Jan2016/Project/Code_from_CT/CT_AuditoryCommand_Level2.csv")
x= c(1,30:38, 41:43)
data = data[,-x]
data_new = data[,-nearZeroVar(data)]
ctrl = trainControl(method = "repeatedcv",repeats = 1) # Define training method
fitlm = train(EndAccuracy ~ ., # Select training feature
              data = data_new, # Select database
              method = "svmRadial", # Select training method svmRadial (linear reg: lm)
              preProcess = c("center","scale"),
              trControl = ctrl,
              tuneLength = 6,
              importance = TRUE) # Define pre-process parameters
summary(fitlm)


################################################
#Predicting end accuracy using random forests
rm(list=ls())
library(caret)

data = read.csv("/Users/Charm/OneDrive/Insight_Work_Jan2016/Project/Code_from_CT/CT_AuditoryCommand_Level2.csv")
x= c(1,30:38, 41:43)
data = data[,-x]
data_new = data[,-nearZeroVar(data)]
ctrl = trainControl(method = "oob",repeats = 5) # Define training method
fitlm = train(EndAccuracy ~ ., # Select training feature
              data = data_new, # Select database
              method = "rf", #random forest
              preProcess = c("center","scale"),
              trControl = ctrl,
              tuneLength = 6,
              importance = TRUE) # Define pre-process parameters
summary(fitlm)
imp = varImp(fitlm)
plot(imp)
imp$importance


################################################
#Predicting plateau accuracy index
rm(list=ls())
library(caret)

data = read.csv("/Users/Charm/OneDrive/Insight_Work_Jan2016/Project/Code_from_CT/CT_AuditoryCommand_Level2.csv")
x= c(1,30:38,40,41,43,44)
data = data[,-x]
data_new = data[,-nearZeroVar(data)]
ctrl = trainControl(method = "repeatedcv",repeats = 1) # Define training method
fitlm = train(PlateauAccuracyInd ~ ., # Select training feature
              data = data_new, # Select database
              method = "lm", # Select training method svmRadial (linear reg: lm)
              preProcess = c("center","scale"),
              trControl = ctrl,
              tuneLength = 6,
              metric = "Rsquared") # Define pre-process parameters
summary(fitlm)
imp = varImp(fitlm)
plot(imp)
imp$importance


################################################
#Predicting response count
rm(list=ls())
library(caret)

data = read.csv("/Users/Charm/OneDrive/Insight_Work_Jan2016/Project/Code_from_CT/CT_AuditoryCommand_Level2.csv")
x= c(1,30:31, 33:38, 41:43)
data = data[,-x]
data_new = data[,-nearZeroVar(data)]
ctrl = trainControl(method = "repeatedcv",repeats = 1) # Define training method
fitlm = train(ResponseCount ~ ., # Select training feature
              data = data_new, # Select database
              method = "lm", # Select training method svmRadial (linear reg: lm)
              preProcess = c("center","scale"),
              trControl = ctrl,
              tuneLength = 6,
              metric = "Rsquared") # Define pre-process parameters
summary(fitlm)
imp = varImp(fitlm)
plot(imp)
imp$importance



################################################
#Predicting plateau accuracy index
rm(list=ls())
library(caret)

data = read.csv("/Users/Charm/OneDrive/Insight_Work_Jan2016/Project/Code_from_CT/CT_AuditoryCommand_Level2.csv")
x= c(1,30:38,40,41,43,44)
data = data[,-x]
data_new = data[,-nearZeroVar(data)]
ctrl = trainControl(method = "repeatedcv",repeats = 1) # Define training method
fitlm = train(PlateauAccuracyInd ~ ., # Select training feature
              data = data_new, # Select database
              method = "lm", # Select training method svmRadial (linear reg: lm)
              preProcess = c("center","scale"),
              trControl = ctrl,
              tuneLength = 6,
              metric = "Rsquared") # Define pre-process parameters
summary(fitlm)
imp = varImp(fitlm)
plot(imp)
imp$importance



###############################################
#VoiceMail Task
#Predicting end accuracy using linear regression
rm(list=ls())
library(caret)

data = read.csv("/Users/Charm/OneDrive/Insight_Work_Jan2016/Project/Code_from_CT/CT_VoiceMailTask_Level2.csv")
x= c(1,30:38,41:43)
data = data[,-x]
data_new = data[,-nearZeroVar(data)]
ctrl = trainControl(method = "repeatedcv",repeats = 1) # Define training method
fitlm = train(EndAccuracy ~ ., # Select training feature
              data = data_new, # Select database
              method = "lm", # Select training method svmRadial (linear reg: lm)
              preProcess = c("center","scale"),
              trControl = ctrl,
              tuneLength = 6,
              metric = "Rsquared") # Define pre-process parameters
summary(fitlm)
imp = varImp(fitlm)
plot(imp)
imp$importance



#Predicting end accuracy using random forests regression
rm(list=ls())
library(caret)

data = read.csv("/Users/Charm/OneDrive/Insight_Work_Jan2016/Project/Code_from_CT/CT_VoiceMailTask_Level2.csv")
x= c(1,30:38, 41:43)
data = data[,-x]
data_new = data[,-nearZeroVar(data)]
ctrl = trainControl(method = "oob",repeats = 1) # Define training method
fitlm = train(EndAccuracy ~ ., # Select training feature
              data = data_new, # Select database
              method = "rf", #random forest
              preProcess = c("center","scale"),
              trControl = ctrl,
              tuneLength = 6,
              importance = TRUE) # Define pre-process parameters
summary(fitlm)
imp = varImp(fitlm)
plot(imp)
imp$importance