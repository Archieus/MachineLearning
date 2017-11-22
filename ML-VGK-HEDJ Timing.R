library(quantmod)
library(mlbench)
library(caret)
library(e1071)

#alphavantage API = Y474

####Download ADJUSTED PRICE DATA from AlphaVantage
###outputsize=c(full,compact) full= 20 years of data, compact = 100 datapoints

VGK.csv <- read.csv('http://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=VGK&outputsize=full&apikey=Y474&datatype=csv')
HEDJ.csv <- read.csv('http://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=HEDJ&outputsize=full&apikey=Y474&datatype=csv')

VGK <- cbind(VGK.csv[,c(2:5,7,6)])
rownames(VGK) <- VGK.csv[,1]
VGK <- as.xts(VGK)

HEDJ <- cbind(HEDJ.csv[,c(2:5,7,6)])
rownames(HEDJ) <- HEDJ.csv[,1]
HEDJ <- as.xts(HEDJ)

Symbols <- c("DEXUSEU", "DTWEXM")
getSymbols(Symbols, src = "FRED", from = "2007-01-01")

#### CONVERT TO MONTHLY DATA ####
DXY0 <- to.weekly(DTWEXM, indexAt = "lastof")
EURUSD <- to.weekly(DEXUSEU, indexAt = "lastof")
HEDJ.wk <- to.weekly(HEDJ, indexAt = "lastof")
VGK.wk <- to.weekly(VGK, indexAt = "lastof")

DXY0.ROC <- ROC(DXY0,1, "discrete", na.pad = FALSE)
EURUSD.ROC <- ROC(EURUSD, 1, "discrete", na.pad = FALSE)
HEDJ.ROC <- ROC(HEDJ.wk, 1, "discrete", na.pad = FALSE)
VGK.ROC <- ROC(VGK.wk, 1, "discrete", na.pad = FALSE)

State <- cbind(ifelse(VGK.ROC[,4] > 0,"Up", "Down"))
State <- as.data.frame(lag(State, -1))


factors <- as.data.frame(cbind(DXY0.ROC[,4], EURUSD.ROC[,4], HEDJ.ROC[,6]))
factors <- na.locf(factors, na.rm = FALSE, fromLast = TRUE)

dataf <- na.omit(cbind(tail(factors, nrow(State)), State))
colnames(dataf) <- c("DXY0ROC", "EURUSDROC", "HEDJROC", "State")

#### CREATED LIST of 80% OF ROWS IN ORIGINAL DATASET TO TRAIN THE MODEL ####
validation_index <- createDataPartition(dataf$State, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- dataf[-validation_index,]
# use the remaining 80% of data to training and testing the models
USDState <- dataf[validation_index,]


#### IDENTIFY BEST FEATURES OF THE LIST FOR THE MODEL ####
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(dataf[,1:3], dataf[,4], sizes=c(1:3), rfeControl=control)
# summarize the results
print(results)

#### IDENTIFY HIGHLY CORRELATED FEATURES TO REMOVE REDUNDANCY ####
correlationMatrix <- round(cor(dataf[,1:3]),4)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

#### SUMMARIZE DATASET ####
# dimensions of dataset
dim(dataf)

# list types for each attribute
sapply(dataf, class)

# list the levels for the class
levels(dataf$State)

# summarize the class distribution
percentage <- prop.table(table(dataf$State)) * 100
cbind(freq=table(dataf$State), percentage=percentage)

summary(dataf)

#### VISUALIZE DATASETS ####
# split input and output
x <- dataf[,1:3]
y <- dataf[,4]

# boxplot for each attribute on one image
par(mfrow=c(1,3))
for(i in 1:3) {
  boxplot(x[,i], main=names(dataf[-4])[i])
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
metric <- "Accuracy"

#### BUILD MODELS ####
# A) LINEAR ALGORITHMS
fit.lda <- train(State~., data=dataf, method="lda", metric=metric, trControl=control)

# B) NONLINEAR ALGORITHMS
# CART
fit.cart <- train(State~., data=dataf, method="rpart", metric=metric, trControl=control)
# kNN
fit.knn <- train(State~., data=dataf, method="knn", metric=metric, trControl=control)

# C) ADVANCED ALGORITHMS
# SVM
fit.svm <- train(State~., data=dataf, method="svmRadial", metric=metric, trControl=control)
# Random Forest
fit.rf <- train(State~., data=dataf, method="rf", metric=metric, trControl=control)

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

#### PREDICTIONS ON LIVE DATA ####
ModelPred <- predict(fit.svm,dataf)
confusionMatrix(ModelPred, dataf$State)
last(ModelPred)
