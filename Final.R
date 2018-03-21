#Set wd

library(mlbench)# load libraries
library(caret)
library(e1071)
mydata1 <- read.csv("database.csv")# load the dataset
mydata <- mydata1
skewness(mydata1$LIMIT_BAL)
histogram(mydata1$LIMIT_BAL)
#find skewness, centre scaling, historgrams

# summarize 
summary(mydata)
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(mydata, method=c("BoxCox"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, mydata)
mydata<-transformed
# summarize the transformed dataset
summary(mydata)
skewness(mydata$LIMIT_BAL)
histogram(mydata$LIMIT_BAL)

# Summarize the dataset
summary(mydata)
mydata$default.payment.next.month <- factor(mydata$default.payment.next.month)
mydata$SEX <- as.factor(mydata$SEX)
mydata$EDUCATION <-  as.factor(mydata$EDUCATION)
mydata$MARRIAGE <- as.factor(mydata$MARRIAGE)
mydata = na.omit(mydata)
summary(mydata)
# Find the number of rows of the dataset
row<-nrow(mydata)
row 
set.seed(12345)
trainindex <- sample(row, 0.7*row, replace=FALSE)
training <- mydata[trainindex,]
validation <- mydata[-trainindex,]

##############################################
trainingx<- training  # segmenting for random forest 
trainingy <- training[,24]            
validationx <- validation[,-24]
validationy <- validation[,24]
##############################################
# Random Forest
install.packages("randomForest")
library(randomForest)
rfModel = randomForest( training$default.payment.next.month ~ ., data=training, ntree=500 ) # ntree=500
varImpPlot(rfModel) # This was done to understand the variable importance plot to aide our feature selection process

#############################################

# Set 'family=binomial' for a logistic regression model

mylogit<-glm(default.payment.next.month ~.,data=training, family=binomial)

# Summary of the regression
summary(mylogit)

# Model coefficients
coef(mylogit)

# stepwise logistic regression with backward selection
mylogit.step = step(mylogit, direction='backward')
summary(mylogit.step)

####################################################################################################################

# Prediciting the validation dataset
mylogit.probs<-predict(mylogit.step,validation,type="response")
mylogit.probs


# Check some of the predicted values in validation set
# Note these predicted values are probabilities
mylogit.probs[1:5]


# Create confusion matrix by specifying a cutoff value
mylogit.pred = rep("No", 0.3*row)
mylogit.pred[mylogit.probs > 0.5] = "Yes"
table(mylogit.pred, validation$default.payment.next.month)




########################################################################

# Next we show how to choose the best threshold value with highest accuary
# creating a range of values to test for accuracy
library(SDMTools)
thresh=seq(0,1,by=0.05)

# Initializing a 1*20 matrix of zeros to save values of accuracy
acc = matrix(0,1,20)

# computing accuracy for different threshold values from 0 to 1 step by 0.05
for (i in 1:21){
  matrix = confusion.matrix(validation$default.payment.next.month,mylogit.probs,threshold=thresh[i])
  acc[i]=(matrix[1,1]+matrix[2,2])/nrow(validation)
}
# print and plot the accuracy vs cutoff threshold values
print(c(accuracy= acc, cutoff = thresh))
plot(thresh,acc,type="l",xlab="Threshold",ylab="Accuracy", main="Validation Accuracy for Different Threshold Values")

# attach mylogit.probs into the validation set, and set response as 1 if mylogit.probs >.5 and 0 otherwise.
mydf <-cbind(validation,mylogit.probs)
mydf$response <- as.factor(ifelse(mydf$mylogit.probs>0.5, 1, 0))

# create the ROC
install.packages("ROCR")
library(ROCR)
library(AppliedPredictiveModeling)
logit_scores <- prediction(predictions=mydf$mylogit.probs, labels=mydf$default.payment.next.month)

#PLOT ROC CURVE
logit_perf <- performance(logit_scores, "tpr", "fpr")

plot(logit_perf,
     main="ROC Curves",
     xlab="1 - Specificity: False Positive Rate",
     ylab="Sensitivity: True Positive Rate",
     col="darkblue",  lwd = 3)
abline(0,1, lty = 300, col = "green",  lwd = 3)
grid(col="aquamarine")


# AREA UNDER THE CURVE
logit_auc <- performance(logit_scores, "auc")
as.numeric(logit_auc@y.values)  ##AUC Value

################################################################################################
#KNN Method
trainingx<- training[,-c(2,3,4,5,24)] # The best predictors only
trainingy <- training[,24]            
validationx <- validation[,-c(2,3,4,5,24)] #The best predictors 
validationy <- validation[,24]

library(AppliedPredictiveModeling)
library(e1071) # misc library including skewness function
library(corrplot)
library(lattice)
library(caret) 

#KNN_MODEL
knnModel = train(x=trainingx, y=trainingy, method="knn",
                 preProc=c("center","scale"),
                 tuneLength=10)

# ploting the accuracy of the model with varying K
plot(knnModel$results$k, knnModel$results$Accuracy, type="o",xlab="# neighbors",ylab="Accuracy", main="KNNs for Credit card default")


## Testing the model against validation data set 
knnPred = predict(knnModel, newdata=validationx)



validation_rf <- predict(knnModel,validationx , type = "prob")[,1]



## The function 'postResample' can be used to get the test set performance values
knnPR = postResample(pred=knnPred, obs=validationy)

knnPR

#checking the best threshlod for knn with K =23
thresh=seq(0,1,by=0.05)
acc = matrix(0,1,20)
for (i in 1:21){
  matrix = confusion.matrix(validation$default.payment.next.month,validation_rf,threshold=thresh[i])
  acc[i]=(matrix[1,1]+matrix[2,2])/nrow(validation)
}
print(c(accuracy= acc, cutoff = thresh))
plot(thresh,acc,type="l",xlab="Threshold",ylab="Accuracy", main="Validation Accuracy for Different Threshold Values")

#classification Matrix

KnnPredict = rep("No", 0.3*row)
KnnPredict[knnPred == 1] = "Yes"
table(KnnPredict, validation$default.payment.next.month)

library(pROC)
#ROC CURVE
##################################################################
knnPred_numeric <- data.frame(knnPred)
knnPred_numeric$knnPred <- as.numeric(knnPred_numeric$knnPred)
rocCurve_knn = roc(response = validation$default.payment.next.month, predictor = validation_rf, levels = rev(levels(validation$default.payment.next.month)))
auc(rocCurve_knn) 
plot(rocCurve_knn, legacy.axes = TRUE)

#####################################
#Boosted tree

# Boosted tree: library(gbm)
#set.seed=(12345)
library(gbm)
mydata_gbm <- mydata
mydata_gbm <- mydata_gbm[-c(2,3,4,5)]
mydata_gbm$default.payment.next.month <- ifelse(mydata_gbm$default.payment.next.month==1,'yes','no')
mydata_gbm$default.payment.next.month <- as.factor(mydata_gbm$default.payment.next.month)

row_gbm<-nrow(mydata_gbm)
row_gbm 
set.seed(12345)
trainindex_gbm <- sample(row_gbm, 0.7*row, replace=FALSE)
training_gbm <- mydata_gbm[trainindex_gbm,]
validation_gbm <- mydata_gbm[-trainindex_gbm,]

trainingx_gbm<- training_gbm[,-20]
trainingy_gbm <- training_gbm[,20]
validationx_gbm <- validation_gbm[,-20]
validationy_gbm <- validation_gbm[,20]


objControl <- trainControl(method='cv', number=3, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)
objModel <- train(trainingx_gbm, trainingy_gbm, 
                  method='gbm', 
                  trControl=objControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"))
summary(objModel)
print(objModel)
gbm_pred <- predict(object=objModel, validationx, type='raw')
gbm_pred

gbmPR = postResample(pred=gbm_pred, obs=validationy_gbm)

gbmPR

gbm_probs <- predict(object=objModel, validationx_gbm, type='prob')[,1]
gbm_probs

#classification Matrix

gbmPredict = rep("No", 0.3*row)
gbmPredict[gbm_pred == "yes"] = "Yes"
table(gbmPredict, validation_gbm$default.payment.next.month)

#RoC curve for gbm
##################################################################
rocCurve_gbm = roc(response = validation_gbm$default.payment.next.month, predictor = gbm_probs, levels = rev(levels(validation_gbm$default.payment.next.month)))
auc(rocCurve_gbm) 
plot(rocCurve_gbm, legacy.axes = TRUE)

