newwd="D:/University/4th Year/ST4035 - Data Science/University-Data-Science-ST4035-Projects/Project-1/Datasets"
setwd(newwd)

#install.packages('dplyr', repos = 'https://cloud.r-project.org')
#install.packages("caTools")
#install.packages("ROCR")
#install.packages("pscl")
set.seed(100)
library(dplyr)
library(caTools)
library(ROCR)
library(pscl)

train=read.csv("D:/University/4th Year/ST4035 - Data Science/University-Data-Science-ST4035-Projects/Project-1/Datasets/train_main.csv")
test=read.csv("D:/University/4th Year/ST4035 - Data Science/University-Data-Science-ST4035-Projects/Project-1/Datasets/test_main.csv")

y = train$Final
y = y - 1
X_train = train %>% select(-Final)
head(train)

summary(X_train)

str(X_train)

count_unique = function(col) {
  return(length(unique(col)))
}

#Categoricas being weeded out
unique_counts = apply(X_train, 2, count_unique)
float_indexes=unique_counts>15
cat_cols=colnames(X_train)[!float_indexes]


X_train[cat_cols] = lapply(X_train[cat_cols], factor)
test[cat_cols] = lapply(test[cat_cols], factor)


#Train test split
ratio = 0.8
train_indices = sample(1:nrow(X_train), size = round(nrow(X_train) * ratio))

# Split the data into training and test sets
train_set <- X_train[train_indices, ]
test_set <- X_train[-train_indices, ]

y_train = y[train_indices]
y_test = y[-train_indices]

train_set$Final=y_train

str(train_set)

#MODEL
model = glm(Final~.,family = binomial, data = train_set)
summary(model)

#Checking for deviance
ANOVA_MODEL_1=anova(model, test="Chisq")

predictions = predict(model, newdata = test_set, type='response')
predictions = ifelse(predictions > 0.5,1,0)

#Checking for misclassification
misClasificError <- mean(predictions != y_test)
print(paste('Accuracy',1-misClasificError))


#New Classification
cols_for_backward_selection=(ANOVA_MODEL_1$`Pr(>Chi)` < 0.05)

replace_na_with_false = function(x) {
  is_na = is.na(x)
  x[is_na]=FALSE
  return(x)
}
replace_na_with_true = function(x) {
  is_na = is.na(x)
  x[is_na]=TRUE
  return(x)
}
cols_for_backward_selection=replace_na_with_true(cols_for_backward_selection)
back_cols_1=colnames(test_set)[cols_for_backward_selection[2:137]]

#2nd iteration
train_set_2=train_set[,cols_for_backward_selection[2:137]]
test_set_2=test_set[,cols_for_backward_selection[2:136]]

#MODEL
model = glm(Final~.,family = binomial, data = train_set_2)
summary(model)

#Checking for deviance
ANOVA_MODEL_2=anova(model, test="Chisq")

predictions = predict(model, newdata = test_set_2, type='response')
predictions = ifelse(predictions > 0.5,1,0)

#Checking for misclassification
misClasificError <- mean(predictions != y_test)
print(paste('Accuracy',1-misClasificError))


cols_for_backward_selection=(ANOVA_MODEL_2$`Pr(>Chi)` < 0.05)
cols_for_backward_selection=replace_na_with_true(cols_for_backward_selection)
back_cols_1=colnames(test_set_2)[cols_for_backward_selection[2:length(cols_for_backward_selection)]]

train_set_3=train_set_2[,cols_for_backward_selection[2:length(cols_for_backward_selection)]]
test_set_3=test_set_2[,cols_for_backward_selection[2:length(cols_for_backward_selection)]]

#MODEL
model = glm(Final~.,family = binomial, data = train_set_3)
summary(model)

#Checking for deviance
ANOVA_MODEL_2=anova(model, test="Chisq")

predictions = predict(model, newdata = test_set_3, type='response')
predictions = ifelse(predictions > 0.5,1,0)

#Checking for misclassification
misClasificError <- mean(predictions != y_test)
print(paste('Accuracy',1-misClasificError))



#Main inference
test_set_main=test[,cols_for_backward_selection[2:length(cols_for_backward_selection)]]
preds = predict(model, newdata = test_set_main, type='response')
preds = ifelse(preds > 0.5,1,0)
preds = preds + 1
sub4=data.frame(
  "ID"=1:length(preds),
  "Final"=preds
)
write.csv(sub4, file = "sub4.csv", row.names = FALSE)
