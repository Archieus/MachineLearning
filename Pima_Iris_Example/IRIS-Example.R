# load caret package
library(caret)
library(mlbench)
library(randomForest)

# load the dataset
data(iris)

#### Prepare For Modeling by Pre-Processing Data ####
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(iris[,1:4], method=c("range"))
# transform the dataset using the pre-processing parameters
transformed <- predict(preprocessParams, iris[,1:4])
# summarize the transformed dataset
summary(transformed)

#### Algorithm Evaluation With Resampling Methods ####
# define training control
trainControl <- trainControl(method="cv", number=10)
# estimate the accuracy of Naive Bayes on the dataset
fit <- train(Species~., data=iris, trControl=trainControl, method="nb")
# summarize the estimated accuracy
print(fit)

#### Improve Accuracy with Algorithm Tuning ####
# define a grid of parameters to search for random forest
grid <- expand.grid(.mtry=c(1,2,3,4,5,6,7,8,10))
# estimate the accuracy of Random Forest on the dataset
fit2 <- train(Species~., data=iris, trControl=trainControl, tuneGrid=grid, method="rf")
# summarize the estimated accuracy
print(fit2)

#### Algorithm Evaluation Metrics ####
# prepare 5-fold cross validation and keep the class probabilities
control <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=mnLogLoss)
# estimate accuracy using LogLoss of the CART algorithm
fit <- train(Species~., data=iris, method="rpart", metric="logLoss", trControl=control)
# display results
print(fit)

#### Finalize And Save Your Model ####
data(iris)
# train random forest model
finalModel <- randomForest(Species~., iris, mtry=2, ntree=2000)
# display the details of the final model
print(finalModel)


