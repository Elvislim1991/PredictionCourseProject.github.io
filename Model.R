library(skimr)
library(dplyr)
library(caret)

temp.train <- tempfile()
temp.test <- tempfile()

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
              destfile = temp.train)
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
              destfile = temp.test)

training <- read.csv(temp.train)
testing <- read.csv(temp.test)

unlink(temp.train)
unlink(temp.test)

# Split training to 80% training and 20% validation
set.seed(1106)
trainIndex <- createDataPartition(training$classe, p=.8, list=FALSE, times = 1)
train <- training[trainIndex, ]
validate <- training[-trainIndex, ]

# Clean data
col_exclude <- c("X", "raw_timestamp_part_1", "raw_timestamp_part_2",
                 "user_name", "cvtd_timestamp", "new_window")

# Convert characters col other that col_exclude to numeric and remove features 
# with near zero variance

train.char <- train %>% select_if(is.character) %>% 
        select(-classe, -user_name:-new_window) %>% 
        sapply(FUN = as.numeric) %>% as.data.frame() %>% cbind(classe=train$classe)

train.clean <- train %>% select(-col_exclude) %>% select_if(is.numeric) %>% 
        cbind(train.char)

train.clean <- train.clean[, -nearZeroVar(train.clean)]

# Compare performance of knnImpute (with nearzerovariance dropped) and 
# dropped all empty columns

clean_data <- function(df){
        train.filter <- df %>% select(-col_exclude)
        train.filter.col <- train.filter[,colSums(is.na(train.filter)) == 0]
        train.filter.col.num <- train.filter.col %>% select_if(is.numeric)
}


train.filter.col.num <- clean_data(train) %>% cbind(classe=train$classe)

# # KNN Impute for clean_data
# 
# preObj <- preProcess(train.clean[, -120], method="knnImpute")
# train.clean.impute <- predict(preObj, train.clean[, -120])
# train.clean.impute <- train.clean.impute %>% cbind(classe=train$classe)
# 
# # Random forest on both datasets
# 
# train_control <- trainControl(method="cv", number=10)
# 
# start.time <- Sys.time()
# model.rf.impute <- train(classe~., data=train.clean.impute, method="rf",
#                          trControl=train_control)
# end.time <- Sys.time()
# 
# print(end.time-start.time)
# 
# start.time <- Sys.time()
# model.rf.naRemove <- train(classe~., data=train.filter.col.num, method="rf",
#                            trControl=train_control)
# end.time <- Sys.time()
# 
# print(end.time-start.time)

test.filter.col.num <- clean_data(testing)

validation.filter.col.num <- clean_data(validate)  %>% cbind(classe=validate$classe)

# model with naRemove get better accuracy for 10K CV (99.8%) with 28mins training time
# model with knn imputed after dropped all near zero variance features
# accuracy 99.7% and training time is 1.2 hours
# Preprocess data
preProc.pca <- preProcess(train.filter.col.num, method = "pca", thresh = 0.95)
train.fit.pca <- predict(preProc.pca, train.filter.col.num)
validate.fit.pca <- predict(preProc.pca, validation.filter.col.num)

# Model training 

train_control <- trainControl(method="cv", number=10)

## random forest
start.time <- Sys.time()

model.rf <- train(classe~., data=train.filter.col.num, method="rf",
                     trControl=train_control)

end.time <- Sys.time()
paste("Time to train random forest")
print(end.time - start.time)

start.time <- Sys.time()

model.gbm <- train(classe~., data=train.filter.col.num, method="gbm", 
                   trControl=train_control, verbose=FALSE)

end.time <- Sys.time()
paste("Time to train gbm")
print(end.time - start.time)

start.time <- Sys.time()

model.knn <- train(classe~., data=train.filter.col.num, method="knn", 
                   trControl=train_control)

end.time <- Sys.time()
paste("Time to train knn classifier")
print(end.time - start.time)

# Stacking all 3 clasifiers

pred.rf <- predict(model.rf, validation.filter.col.num)

pred.gbm <- predict(model.gbm, validation.filter.col.num)

pred.knn <- predict(model.knn, validation.filter.col.num)

predDF <- data.frame(pred.rf, pred.gbm, pred.knn, classe=validation.filter.col.num$classe)

comModFit <- train(classe~., method="rf", data=predDF)

combPred <- predict(comModFit, predDF)

# Compare predict results for validation datasets

confusionMatrix(pred.rf, factor(validation.filter.col.num$classe))

confusionMatrix(pred.gbm, factor(validation.filter.col.num$classe))

confusionMatrix(pred.knn, factor(validation.filter.col.num$classe))

confusionMatrix(combPred, factor(validation.filter.col.num$classe))

# Predict on test set
pred.rf.test <- predict(model.rf, test.filter.col.num)

pred.gbm.test <- predict(model.gbm, test.filter.col.num)

pred.knn.test <- predict(model.knn, test.filter.col.num)

pred.test.DF <- data.frame(pred.rf=pred.rf.test, pred.gbm=pred.gbm.test,
                        pred.knn=pred.knn.test)

combPred.test <- predict(comModFit, pred.test.DF)
