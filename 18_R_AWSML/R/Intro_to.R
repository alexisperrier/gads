# ---------------------------------
# Working Directory
# ---------------------------------

# get working directory
getwd()
# set working directory
setwd()

# ---------------------------------
# Packages:
# ---------------------------------

installed.packages()

install.packages('forecast')
library('forecast')

# ---------------------------------
# help:
# ---------------------------------

?c
?forecast

# ---------------------------------
# Convention:
#  x <- 1
# instead of
# x = 1
# I know weird, but you get used to it
# ---------------------------------

a <- c(1,2,3,4,5)
vector <- c(apple = 1, banana = 2, "kiwi fruit" = 3, 4)

# ---------------------------------
# Yeah! Dataframes!
# ---------------------------------

df <- read.csv('iris.csv')


dim(df)
head(df)
colnames(df)
summary(df)

# ---------------------------------
# plot out of the box
# ---------------------------------


plot(df)
hist(df$sepal_length)
boxplot(df$sepal_width)
plot(df$sepal_length, df$petal_width)

# QQ norm
qqnorm(df$sepal_length)
qqline(df$sepal_length)

# ---------------------------------
# ML on Iris
# ---------------------------------

# manual encoding
iris$target[iris$Species == 'setosa'] <- 1
iris$target[iris$Species == 'versicolor'] <- 2
iris$target[iris$Species == 'virginica'] <- 3

# fit model
# notice lm is already available

fit <- lm(target ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = iris )
coefficients(fit)
summary(fit)


