library(Quandl)
library(quantmod)
library(tseries)
library(CausalImpact)
library(mlbench)
library(caret)
library(e1071)

Sys.setenv(TZ = "UTC")
Quandl.api_key("LDvEJuPkQWrPGpHwokVx")

z <- read.csv('http://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=EMB&outputsize=full&apikey=Y474&datatype=csv')

z.new <- cbind(z[,c(2:5,7,6)])
rownames(z.new) <-z[,1]
EMB <- as.xts(z.new)

AUDUSD <- Quandl("RBA/FXRUSD", type = "xts", start_date = "2013-09-09")
USD <- Quandl("CHRIS/ICE_DX1", type = "xts", start_date = "2013-09-09")
VIX <- Quandl("CBOE/VIX", type = "xts", start_date = "2007-01-03")
getSymbols('BAMLH0A0HYM2', src  = 'FRED', return.xlass = 'xts', from = '1996-12-31')

#### CREATE DATA FOR MACHINE LEARNING PREDICTION ####
EMB.mo <- to.monthly(EMB[,6], indeAt = 'firstof')
EMB.ROC <- ROC(EMB.mo,1)

HYOAS.mo <- to.monthly(na.locf(BAMLH0A0HYM2, na.rm = FALSE, fromLast = TRUE), indexAt='firstof')
VIX.mo <- to.monthly(na.locf(VIX[,4], na.rm = FALSE, fromLast = TRUE), indexAt = 'firstof')
AUDUSD.mo <- to.monthly(AUDUSD[,1], indexAt = 'firstof')
USD.mo <- to.monthly(USD[,4], indexAt = 'firstof')

State <- cbind(ifelse(EMB.ROC[,4] > 0,"Up", "Down"))
State <- lag(State, -1)

Live <- State
State <- as.data.frame(State)
Live.df <- as.data.frame(Live)
Live.df <- na.omit(Live.df)

RawData <- as.data.frame(na.omit(cbind(HYOAS.mo[,4],VIX.mo[,4],AUDUSD.mo[,4], USD.mo[,4])))

LDRoCt <- ifelse(nrow(RawData) > nrow(Live.df),nrow(Live.df),nrow(RawData))

LiveData <- na.omit(cbind(tail(RawData, LDRoCt), tail(Live.df, LDRoCt)))
names(LiveData) <- c("HYSprd","VIX","AUDUSD", "USD","State")

USRow <- ifelse(nrow(RawData) > nrow(State), nrow(State), nrow(RawData))

USDData <- na.omit(cbind(tail(RawData, USRow), tail(State, USRow)))
names(USDData) <- c("HYSprd","VIX","AUDUSD", "USD","State")

#### IDENTIFY BEST FEATURES OF THE LIST FOR THE MODEL ####

control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(USDData[,1:4], USDData[,5], sizes=c(1:4), rfeControl=control)
# summarize the results
print(results)

#### IDENTIFY HIGHLY CORRELATED FEATURES TO REMOVE REDUNDANCY ####
correlationMatrix <- round(cor(USDData[,1:4]),4)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

#### CREATED LIST of 80% OF ROWS IN ORIGINAL DATASET TO TRAIN THE MODEL ####
validation_index <- createDataPartition(USDData$State, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- USDData[-validation_index,]
# use the remaining 80% of data to training and testing the models
USDData <- USDData[validation_index,]

#### SUMMARIZE DATASET ####
# dimensions of dataset
dim(USDData)

# list types for each attribute
sapply(USDData, class)

# list the levels for the class
levels(USDData$State)

# summarize the class distribution
percentage <- prop.table(table(USDData$State)) * 100
cbind(freq=table(USDData$State), percentage=percentage)

summary(USDData)

#### VISUALIZE DATASETS ####
# split input and output
x <- USDData[,1:4]
y <- USDData[,5]

# boxplot for each attribute on one image
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(USDData[-5])[i])
}

# barplot for class breakdown
plot(y)

# scatterplot matrix (requires ellipse package to be available)
featurePlot(x=x, y=y, plot="ellipse")

# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")

# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

#### EVALUATE SOME ALGORITHMS ####
# Run algorithms using 10-fold cross validation
# if method = "repeadedcv", "repeats =" can be used
# repeats can be dangerous if model is trained in "deteministic manner"
control <- trainControl(method="cv", number=10)

#### CONTROL METHODS ####
#### Rolling Window aka Forward Chainging / Walk Forward Analysis best for time series analysis ####
## Resampling Methods in Caret ##
## "boot", "boot632", "optimism_boot", "boot_all", "cv", "repeatedcv", "LOOCV", "LGOCV" ##
## "none" (only fits one model to the entire training set),
## "oob" (only for random forest, bagged trees, bagged earth, bagged flexible discriminant analysis,
## or conditional tree forest models)
## timeslice, "adaptive_cv", "adaptive_boot" or "adaptive_LGOCV"

#control <- trainControl(method = "timeslice", initialWindow = 97,  fixedWindow = TRUE, horizon = 1)

####
###Metric "RMSE" & "Rsquared" for Regression; "Accuracy" & "Kappa" for Classification
metric <- "Accuracy"

#### BUILD MODELS ####
# A) LINEAR ALGORITHMS
fit.lda <- train(State~., data=USDData, method="lda", metric=metric, trControl=control)

# B) NONLINEAR ALGORITHMS
# CART
fit.cart <- train(State~., data=USDData, method="rpart", metric=metric, trControl=control)
# kNN
fit.knn <- train(State~., data=USDData, method="knn", metric=metric, trControl=control)

# C) ADVANCED ALGORITHMS
# SVM
fit.svm <- train(State~., data=USDData, method="svmRadial", metric=metric, trControl=control)
# Random Forest
fit.rf <- train(State~., data=USDData, method="rf", metric=metric, trControl=control)

#### SELECT BEST MODEL ####
# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.svm)

#### MAKE PREDICTIONS USING BEST MODEL ####
# estimate skill of the Model on the validation dataset
predictions <- predict(fit.svm, validation)
confusionMatrix(predictions, validation$State)

########## FINALIZE MODEL ##########
# Save Final Model #
final_model <- svm(State~., LiveData) ###The entire Dataset
# save the model to disk
saveRDS(final_model, "./USD_model.rds")

# load the model
super_model <- readRDS("./USD_model.rds")
print(super_model)

### Predictive Model with probabilities ###
pred.model <- svm(State~., LiveData, probability = TRUE)
predict(pred.model, LiveData[,1:4], probability = TRUE)

KNN_Final_Model <- knnreg(State~., LiveData)
predict(KNN_Final_Model, LiveData[,1:4])

# make a predictions on "new data" using the final model
final_predictions <- predict(super_model, LiveData[,1:4])
confusionMatrix(final_predictions, LiveData$State)
###################### END MACHINE LEARNING PREDICTIONS #####################

