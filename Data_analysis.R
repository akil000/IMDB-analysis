# ======= Step 1 - data collection =======

# information about dataset on site:
# https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
# http://ai.stanford.edu/~amaas/data/sentiment/

# ======= Functions needed for plotting, etc.=======
library(caret)
## represents a cross table
confusion_plot <- function(cross_table) {
  plt <- as.data.frame(cross_table)
  plt$t.x <- factor(plt$t.x, levels = levels(plt$t.x))

  {
    ggplot(plt, aes(t.y, t.x, fill = t.Freq)) +
      geom_tile() +
      geom_text(aes(label = t.Freq), size = 6) +
      scale_fill_gradient(low = "lightblue", high = "firebrick") +
      labs(x = "Labels", y = "Predictions", fill = "") +
      scale_x_discrete(labels = c("negative", "positive")) +
      scale_y_discrete(labels = c("negative", "positive")) +
      theme(
        panel.background = element_blank(),
        axis.text = element_text(size = 14)
      )
  }
}

## Appearance of the sample plot
appearance <- theme(
  plot.background = element_rect(fill = "#F7FFDD"),
  panel.background = element_rect(fill = "#DAD8D5"),
  plot.caption = element_text(color = "#99CCFF"),
  plot.title = element_text(face = "bold", size = (14), color = "steelblue")
)

## Function for comparing the count of positive and negative reviews on a plot

data_proportion_plot <- function(data, sentiment) {
  ggplot(data = data, aes(sentiment, fill = sentiment)) +
    geom_bar() +
    ggtitle("Count of positive and negative reviews") +
    ylab("Count") +
    appearance
}

## It can represent, for example: the count of reviews based on a given parameter such as: review length, number of punctuation marks

parameters_review_plot <- function(data, parameter, title) {
  ggplot(data = data, aes(x = parameter, fill = factor(sentiment))) +
    geom_bar(position = "dodge") +
    appearance +
    theme(
      panel.background = element_blank(),
      legend.title = element_blank()
    ) +
    labs(title = title)
}


## Review length
count_words <- function(data) {
  Review_length <- NULL
  i <- 1
  for (i in 1:length(data[, 1])) {
    Review_length[i] <- lapply(strsplit(data$review[[i]], " "), length)
  }
  return(Review_length)
}



# ====== Step 2 - Exploration and Data Preparation ======

# setwd("file_path")
# Importing data into a Data Frame

imdb <- read.csv("IMDB Dataset.csv", stringsAsFactors = FALSE)
imdb <- imdb[!duplicated(imdb), ]
row.names(imdb) <- NULL ### Resetting indexes

# Keep the first 10,000
imdb <- imdb[1:10000, ]

imdb_copy <- imdb[, ]
imdb_copy$Review_length <- unlist(count_words(imdb))

# structure
str(imdb)

# Convert positive/negative to factor
imdb$sentiment <- factor(imdb$sentiment)

# Check What Has Changed
str(imdb$sentiment)
table(imdb$sentiment)


# budowanie korpusu
library(tm)
imdb_corpus <- VCorpus(VectorSource(imdb$review))

print(imdb_corpus)
inspect(imdb_corpus[1:2])

as.character(imdb_corpus[[10]])

## Cleaning corpus
# Function useful for removing specific items (i.e., x)
toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))

# Lowercase letters
imdb_corpus_clean <- tm_map(imdb_corpus, content_transformer(tolower))
as.character(imdb_corpus_clean[[10]])

# Removing numbers
imdb_corpus_clean <- tm_map(imdb_corpus_clean, removeNumbers)
as.character(imdb_corpus[[12]])
as.character(imdb_corpus_clean[[12]])

# Stopwords 
imdb_corpus_clean <- tm_map(imdb_corpus_clean, removeWords, stopwords())
as.character(imdb_corpus[[10]])
as.character(imdb_corpus_clean[[10]])

# Removing HTML tags
imdb_corpus_clean <- tm_map(imdb_corpus_clean, toSpace, "</?[^>]+>")
as.character(imdb_corpus[[10]])
as.character(imdb_corpus_clean[[10]])

# Removing interpunctation
imdb_corpus_clean <- tm_map(imdb_corpus_clean, toSpace, "[[:punct:]]+")
as.character(imdb_corpus[[10]])
as.character(imdb_corpus_clean[[10]])

# Stemming
library(SnowballC)
imdb_corpus_clean <- tm_map(imdb_corpus_clean, stemDocument)
as.character(imdb_corpus[[10]])
as.character(imdb_corpus_clean[[10]])

# Removing whitespaces
imdb_corpus_clean <- tm_map(imdb_corpus_clean, stripWhitespace)
as.character(imdb_corpus[[10]])
as.character(imdb_corpus_clean[[10]])


## Comparing changes
as.character(imdb_corpus[[68]])
as.character(imdb_corpus_clean[[68]])

as.character(imdb_corpus[[85]])
as.character(imdb_corpus_clean[[85]])


## Text Tokenization: Splitting into Words
# Creating DTM matrix
imdb_dtm <- DocumentTermMatrix(imdb_corpus_clean)


## Creating training and test sets with a 3:1 ratio
set.seed(123) ###
train_sample <- sample(length(imdb$review), floor(length(imdb$review) * 0.75))

imdb_dtm_train <- imdb_dtm[train_sample, ]
imdb_dtm_test <- imdb_dtm[-train_sample, ]

## Labels for training and test sets
imdb_train_labels <- imdb[train_sample, ]$sentiment
imdb_test_labels <- imdb[-train_sample, ]$sentiment

## Checking if the proportions of positive/negative reviews are similar in training and test sets
prop.table(table(imdb_train_labels))
prop.table(table(imdb_test_labels))


## Word cloud
library(wordcloud)
wordcloud(imdb_corpus_clean,
  max.words = 50,
  random.order = FALSE, rot.per = 0, colors = brewer.pal(10, "Dark2")
)

# Comparing positive and negative reviews

positive <- subset(imdb_corpus_clean, imdb$sentiment == "positive")
negative <- subset(imdb_corpus_clean, imdb$sentiment == "negative")

wordcloud(positive,
  max.words = 50,
  random.order = FALSE, rot.per = 0, colors = brewer.pal(10, "Dark2")
)
wordcloud(negative,
  max.words = 50,
  random.order = FALSE, rot.per = 0, colors = brewer.pal(10, "Dark2")
)

## Words that appear in at least 20 reviews in your training dataset,
imdb_freq_words <- findFreqTerms(imdb_dtm_train, 20)
str(imdb_freq_words)

## Document-Term matrix (DTM) with only the selected words
imdb_dtm_freq_train <- imdb_dtm_train[, imdb_freq_words]
imdb_dtm_freq_test <- imdb_dtm_test[, imdb_freq_words]

# Convert the count of word occurrences into binary values ("Yes" or "No") 
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
  return(x)
}

imdb_train <- apply(imdb_dtm_freq_train, MARGIN = 2, convert_counts)
imdb_test <- apply(imdb_dtm_freq_test, MARGIN = 2, convert_counts)



# ====== Step 3 - Building a model ======
library(e1071)

##Initial model
imdb_classifier_initial <- naiveBayes(imdb_train, imdb_train_labels, laplace = 0)
imdb_test_pred_initial <- predict(imdb_classifier_initial, imdb_test)

library(gmodels)
cross_table_initial <- CrossTable(imdb_test_pred_initial, imdb_test_labels,
                          prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
                          dnn = c("predicted", "actual")
)

confusion_plot(cross_table_initial)

## Refined model
imdb_classifier <- naiveBayes(imdb_train, imdb_train_labels, laplace = 1)

# ====== Step 4 - Evaluating a model ======
imdb_test_pred <- predict(imdb_classifier, imdb_test)

cross_table <- CrossTable(imdb_test_pred, imdb_test_labels,
  prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
  dnn = c("predicted", "actual")
)

confusion_plot(cross_table)

## Show the accuracy of a model 
confusionMatrix(imdb_test_pred, imdb_test_labels)["overall"]


# ====== Plots ======
## Plot showing the relationship between review length and frequency of occurrence (here limited to review lengths of up to 300 words)
library(dplyr)
title <- "Relationship between review length and frequency of occurrence"
parameters_review_plot(
  imdb_copy %>% filter(Review_length < 300),
  (imdb_copy %>% filter(Review_length < 300))$Review_length, title
)

## data proportion
data_proportion_plot(imdb_copy, sentiment)

## results matrix
confusion_plot(cross_table = cross_table)


# ====== Step 5 - Model Refinement ======

## Three things can be adjusted:
# - Laplace smoothing parameter (default is 0)
# - Amount of data (more data -> better model?)
# - Modification of "frequency" i.e., the minimum number of reviews in which each word appears
#   (default for 10,000 data points is parameter 10, which is a ratio of 1/1000)


### 1. Frequency for 10,000 data points, with the number of words in the training set in parentheses
# 10 (7,171 words): 44.2 + 40.7 = 84.9 %
#  5 (11,110 words): 44.9 + 40.5 = 85.4 %
# 15 (5,533 words): 44.0 + 41.0 = 85.0 %
# 20 (4,516 words): 44.1 + 41.0 = 85.1 %
# 25 (3,939 words): 43.8 + 41.1 = 84.9 %
# 30 (3,472 words): 43.6 + 41.2 = 84.8 %
# COMMENT: Freq 20 (calculates much faster than 5)

### 2. Laplace parameter for 10,000 data points (Freq = 20)
# Laplace =   0 : 44.1 + 41.0 = 85.1 %
# Laplace = 0.5 : 44.1 + 41.0 = 85.1 %
# Laplace =   1 : 44.2 + 41.0 = 85.2 %
# Laplace =   5 : 44.8 + 40.3 = 85.1 %
# Laplace =  10 : 45.6 + 39.3 = 84.9 %
# COMMENT: No significant improvement, everything around 85 %, possibly Laplace: 1

### 3. Amount of data (Laplace 0, Freq same percentage of messages)
# Data: 10,000, Freq 20 : 44.1 + 41.0 = 85.1 %
# Data: 20,000, Freq 40 :  42.9 + 42.8 = 85.7 %
# Data: 50,000, Freq 100 : 41.8 + 42.4 = 84.2 %
# COMMENT: No significant improvement, increase at 20,000 due to different data


### SELECTION: 10,000, Laplace: 1, Freq: 20
# Result: 44.2 + 41.0 = 85.2 %



# ====== Decision Trees ======

### WITHOUT BOOSTING - 78.92 %

### Parameter trials for 10,000 data points
# trials = 8  : 81.6 %
# trials = 10 : 80.6 %
# trials = 12 : 81.7 %
# trials = 13 : 82.2 %
# trials = 14 : 82.3 %
# trials = 15 : 82.1 %
# COMMENT: Accuracy increases with the trials parameter

## Model for testing
library(C50)

tree <- C5.0(imdb_train, imdb_train_labels, trials = 8)
summary(tree)
prediction <- predict(tree, imdb_test)
confusionMatrix(prediction, imdb_test_labels)$overall["Accuracy"]

### Testing the model for trial values 16:20
tree_models <- list()
pred_tree <- list()
cross_table_tree <- list()
accuracy <- NULL

for (i in 16:20) {
  tree_models[[i - 15]] <- C5.0(imdb_train, imdb_train_labels, trials = i)
  pred_tree[[i - 15]] <- predict(tree_models[[i - 15]], imdb_test)
  accuracy[i - 15] <- confusionMatrix(pred_tree[[i - 15]], imdb_test_labels)$overall["Accuracy"]
}

accuracy
accuracy_without_loading_tree_model <- c(0.8272, 0.8216, 0.8276, 0.8284, 0.8284)

# Results do not exceed 83%

## "Best" (the best result on the test does not necessarily mean this parameter is the best)
## The best choice seems to be choosing trials around 14

cross_table_tree <- CrossTable(pred_tree[[2]], imdb_test_labels,
  prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
  dnn = c("predicted", "actual")
)

confusion_plot(cross_table_tree)

## Data frame with results for other trials parameters 16:20
trials_scores <- data.frame(trials = seq(16, 20), accuracy = accuracy_without_loading_tree_model)