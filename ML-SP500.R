library(quantmod)
library(mlbench)
library(caret)

Symbols <- c("CPIAUCSL", "UMCSENT", "SP500", "VXVCLS", "FEDFUNDS", "GS10", "T10Y2YM", "UNRATE")
getSymbols(Symbols, src = "FRED", from = "2007-01-01")

SP500ret <- periodReturn(SP500, "monthly", type = "log", indexAt = "startof")
SP500Vol <- periodReturn(VXVCLS, "monthly", type = "arithmetic", indexAt = "startof")
CPIChge <- periodReturn(CPIAUCSL, "monthly", type = "arithmetic")

modData <- na.omit(cbind(SP500ret, SP500Vol, CPIChge, FEDFUNDS, GS10, T10Y2YM, UNRATE))
Data.df <- as.data.frame(modData)
colnames(Data.df) <- c("SP500", "VOl", "CPI", "FEDFUNDS", "10YCMS", "T10Y2YM", "UNRATE")

correlationMatrix <- cor(Data.df[,2:7])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes (Identifies columns to be removed)
print(highlyCorrelated)

# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(SP500~., data=Data.df, method="glm", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

##### FEATURE SELECTION #####
# define the control using a random forest selection function
control2 <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(Data.df[,2:7], Data.df[,1], sizes=c(1:7), rfeControl=control2)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))

