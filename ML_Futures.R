library(quantmod)
library(mlbench)
library(caret)
library(e1071)
library(Quandl)
Quandl.api_key("LDvEJuPkQWrPGpHwokVx")

FF <- Quandl("CHRIS/CME_FF1", type = "xts", start_date = '2007-01-01')
USTBnd <- Quandl("CHRIS/CME_US1", type = "xts", start_date = '2007-01-01')
USTNote <- Quandl("CHRIS/CME_TN1", type = "xts", start_date = '2007-01-01')

Symbols <- c("SP500", "VXVCLS", "T10Y2Y", "TEDRATE")
getSymbols(Symbols, src = "FRED", from = "2007-01-01")

FF.wk <- to.weekly(FF[,6])[,4]
TBd.wk <- to.weekly(USTBnd[,6])[,4]
TNt.wk <- to.weekly(USTNote[,6])[,4]
Vol.wk <- to.weekly(VXVCLS)[,4]
Sprd <- to.weekly(T10Y2Y)[,4]
TED.wk <- to.weekly(TEDRATE)[,4]

Features <- na.omit(cbind(FF.wk, TBd.wk, TNt.wk, Vol.wk, Sprd, TED.wk))

dataset <- as.data.frame(Features)

SP500.ret <- na.locf(weeklyReturn(SP500), na.rm = FALSE, fromLast = TRUE)
State <- cbind(ifelse(SP500.ret > 0,"Up","Down"))
StateLag<- lag(State, -1)

State <- na.omit(as.data.frame(State))
State <- tail(State,nrow(dataset))

dataset <- na.omit(cbind(dataset, State))

names(dataset) <- c("FedFds", "USTBnd", "USTNote", "Vol", "Spread", "TED", "State")

#### IDENTIFY BEST FEATURES OF THE LIST FOR THE MODEL ####

control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(dataset[,1:6], dataset[,7], sizes=c(1:6), rfeControl=control)
# summarize the results
print(results)

#### IDENTIFY HIGHLY CORRELATED FEATURES TO REMOVE REDUNDANCY ####
correlationMatrix <- round(cor(dataset[,1:6]),4)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)
