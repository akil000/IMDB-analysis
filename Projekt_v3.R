#======= Krok 1 - zbieranie danych =======

# informacje na stronie
# https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
# http://ai.stanford.edu/~amaas/data/sentiment/

#======= Funkcje potrzebne do rysunków itp =======
library(caret)
## przedstawienie cross table
confusion_plot = function(cross_table){
plt <- as.data.frame(cross_table)
plt$t.x <- factor(plt$t.x, levels= levels(plt$t.x))

ggplot(plt, aes(t.y, t.x, fill= t.Freq)) +
  geom_tile() + geom_text(aes(label=t.Freq), size = 6) +
  scale_fill_gradient(low="lightblue", high="firebrick") +
  labs(x = "Prawdziwe etykiety",y = "Predykcje", fill = "") +
  scale_x_discrete(labels=c("negatywna","pozytywna")) +
  scale_y_discrete(labels=c("negatywna","pozytywna"))+
  theme(panel.background = element_blank(),
              axis.text = element_text(size = 14))
}

## wyglad przykladowego rysunku
wyglad <- theme(plot.background = element_rect(fill = "#F7FFDD"),
                panel.background = element_rect(fill = '#DAD8D5'),
                plot.caption = element_text(color = "#99CCFF"),
                plot.title = element_text(face = "bold", size = (14),color="steelblue"))

##funkcja do porównania na wykresie liczebności recezji pozytywnych i negatywnych

data_proportion_plot = function(data, sentiment){
ggplot(data = data, aes(sentiment, fill = sentiment))+
  geom_bar()+
  ggtitle("Ilość recenzji negatywnych i pozytywnych")+
  ylab("Ilość")+wyglad
}


## przedstawiać może np: liczebność recenzji o danym parametrze np:dlugosc recenzji, liczba znaków interpunkcyjnych
parameters_review_plot = function(data, parameter, title){
    ggplot(data = data, aes(x = parameter, fill = factor(sentiment)))+
    geom_bar(position = "dodge")+
    wyglad+
    theme(panel.background = element_blank(),
          legend.title = element_blank())+
    labs(title =title)
}


## długość recenzji
count_words = function(data){
  dlugosc_opinii <- NULL
  i=1
  for (i in 1:length(data[, 1])){
    dlugosc_opinii[i] <- lapply(strsplit(data$review[[i]],' '), length)
  }
  return(dlugosc_opinii)
}


#====== Krok 2 - eksploracja i przygotowanie danych ======

## Korzystamy z 10 000 recenzji, na koniec będzie to skomentowane

#setwd("ścieżka do folderu") 
# importowanie danych do ramki
imdb = read.csv("IMDB Dataset.csv", stringsAsFactors = FALSE)
imdb = imdb[!duplicated(imdb), ]
row.names(imdb) <- NULL ### reset indeksów

# pierwsze 10 000 zostawiamy
imdb = imdb[1:10000,]

# przyda się do rysunków
imdb_copy =  imdb[ , ]
imdb_copy$dlugosc_recenzji <- unlist(count_words(imdb))

# struktura zbioru
str(imdb)

# konwersja positive / negative na factor
imdb$sentiment = factor(imdb$sentiment)

# sprawdzamy co się zmieniło
str(imdb$sentiment)
table(imdb$sentiment)


# budowanie korpusu
library(tm)
imdb_corpus = VCorpus(VectorSource(imdb$review))

print(imdb_corpus)
inspect(imdb_corpus[1:2])

as.character(imdb_corpus[[10]])

## czyszczenie korpusu
# funcja, która przyda się do usuwania konkretnych rzeczy (czyli x)
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))

# małe litery
imdb_corpus_clean = tm_map(imdb_corpus, content_transformer(tolower))
as.character(imdb_corpus_clean[[10]])

# usunięcie liczb
imdb_corpus_clean = tm_map(imdb_corpus_clean, removeNumbers)
as.character(imdb_corpus[[12]])
as.character(imdb_corpus_clean[[12]])

# stopwords czyli najczęściej używane
imdb_corpus_clean = tm_map(imdb_corpus_clean, removeWords, stopwords())
as.character(imdb_corpus[[10]])
as.character(imdb_corpus_clean[[10]])

# usunięcie znaków HTML czyli u nas przeważnie <br />
imdb_corpus_clean = tm_map(imdb_corpus_clean, toSpace, "</?[^>]+>")
as.character(imdb_corpus[[10]])
as.character(imdb_corpus_clean[[10]])

# usuwanie interpunkcji, własna funkcja, a nie removePunctuation
imdb_corpus_clean = tm_map(imdb_corpus_clean, toSpace, "[[:punct:]]+")
as.character(imdb_corpus[[10]])
as.character(imdb_corpus_clean[[10]])

# redukcja do rdzenia słów
library(SnowballC)
imdb_corpus_clean = tm_map(imdb_corpus_clean, stemDocument)
as.character(imdb_corpus[[10]])
as.character(imdb_corpus_clean[[10]])

# usuwanie zbędnych białych znaków
imdb_corpus_clean = tm_map(imdb_corpus_clean, stripWhitespace)
as.character(imdb_corpus[[10]])
as.character(imdb_corpus_clean[[10]])


## porównanie przykładów przed i po
as.character(imdb_corpus[[68]])
as.character(imdb_corpus_clean[[68]])

as.character(imdb_corpus[[85]])
as.character(imdb_corpus_clean[[85]])


## tokenizacja tekstu, podział na wyrazy
# tworzenie macierzy DTM
imdb_dtm = DocumentTermMatrix(imdb_corpus_clean)


## tworzenie zbioru uczącego i testowego w stosunku 3:1
set.seed(123) ###
train_sample = sample(length(imdb$review), floor(length(imdb$review) *0.75) )

imdb_dtm_train = imdb_dtm[train_sample,]
imdb_dtm_test = imdb_dtm[-train_sample,]

## etykiety dla zbriorów uczącego i testowego
imdb_train_labels = imdb[train_sample,]$sentiment
imdb_test_labels = imdb[-train_sample,]$sentiment

## sprawdzamy czy proporcje recenzji poz/neg są podobne w test i train
prop.table(table(imdb_train_labels))
prop.table(table(imdb_test_labels))


## chmura wyrazów
library(wordcloud)
wordcloud(imdb_corpus_clean, max.words = 50, 
          random.order=FALSE, rot.per=0 ,colors=brewer.pal(10, "Dark2"))

# porównanie positive i negative

positive = subset(imdb_corpus_clean, imdb$sentiment=="positive")
negative = subset(imdb_corpus_clean, imdb$sentiment=="negative")

wordcloud(positive, max.words = 50,
          random.order=FALSE, rot.per=0 ,colors=brewer.pal(10, "Dark2"))
wordcloud(negative, max.words = 50,
          random.order=FALSE, rot.per=0 ,colors=brewer.pal(10, "Dark2"))


## słowa, które wystąpiły co najmniej w 20 recenzjach w naszym zbiorze uczącym
imdb_freq_words = findFreqTerms(imdb_dtm_train, 20)
str(imdb_freq_words)

## macierz DTM tylko z tymi słowami
imdb_dtm_freq_train = imdb_dtm_train[, imdb_freq_words]
imdb_dtm_freq_test = imdb_dtm_test[, imdb_freq_words]

# konwersja liczby wystapien wyrazu na yes lub no
convert_counts = function(x){
  x = ifelse(x>0, "Yes", "No")
  return(x)
}

imdb_train = apply(imdb_dtm_freq_train, MARGIN = 2, convert_counts)
imdb_test = apply(imdb_dtm_freq_test, MARGIN = 2, convert_counts)



#====== Krok 3 - budowa modelu ======
library(e1071)
imdb_classifier = naiveBayes(imdb_train, imdb_train_labels, laplace = 1)



#====== Krok 4 - ocena modelu======
imdb_test_pred = predict(imdb_classifier, imdb_test)

library(gmodels)
cross_table = CrossTable(imdb_test_pred, imdb_test_labels,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

confusion_plot(cross_table)

## pokazuje m.in accuracy modelu
confusionMatrix(imdb_test_pred, imdb_test_labels)['overall']


#====================== rysunki =============================================

##"Wykres zależnosci między długościa recenzji, a częstotliwości wystepowania"(tutaj ograniczyłem sie do długości 300 słów)
library(dplyr)
title = "Wykres zależnosci między długościa recenzji, a częstotliwości występowania"
parameters_review_plot(imdb_copy %>% filter(dlugosc_recenzji < 300), 
                       (imdb_copy %>% filter(dlugosc_recenzji < 300))$dlugosc_recenzji,title)
    
## proporcja danych
data_proportion_plot(imdb_copy, sentiment)

## macierz wyników
confusion_plot(cross_table = cross_table)



#====== Krok 5 - dopracowanie  modelu======

## Można zmieniać 3 rzeczy:
# - parametr wygładzania Laplace (domyślnie jest 0)
# - liczba danych (więcej danych -> lepszy model?)
# - modyfikacja "frequency" czyli w min ilu recenzjach pojawia się każde słowo
#   (domyślnie dla 10 000 dajemy parametr 10 czyli proporcja 1/1000)


### 1. Frequency dla ilości danych 10 000, w nawiasie liczba słów w treningowym
# 10 (7171  słów): 44.2 + 40.7 = 84.9 %   OK
#  5 (11110 słów): 44.9 + 40.5 = 85.4 %   OK
# 15 (5533  słów): 44.0 + 41.0 = 85.0 %   OK
# 20 (4516  słów): 44.1 + 41.0 = 85.1 %   OK   
# 25 (3939  słów): 43.8 + 41.1 = 84.9 %   OK
# 30 (3472  słów): 43.6 + 41.2 = 84.8 %   OK
# KOMENTARZ: Freq 20 (liczy się dużo szybciej niż 5) 

### 2, Parametr Laplace dla 10 000 danych (Freq = 20)
# Laplace =   0 : 44.1 + 41.0 = 85.1 %    OK     
# Laplace = 0.5 : 44.1 + 41.0 = 85.1 %    OK
# Laplace =   1 : 44.2 + 41.0 = 85.2 %    OK
# Laplace =   5 : 44.8 + 40.3 = 85.1 %    OK
# Laplace =  10 : 45.6 + 39.3 = 84.9 %    OK
# KOMENTARZ: Nic znacznie lepszego, wszystko ok 85 %, ewentualnie może Laplace: 1

### 3, Liczba danych (Laplace 0, Freq ten sam procent wiadomości)
# Dane: 10 000, Freq  20 : 44.1 + 41.0 = 85.1 %     OK
# Dane: 20 000, Freq  40 : 42.9 + 42.8 = 85.7 %     OK
# Dane: 50 000, Freq 100 : 41.8 + 42.4 = 84.2 %     OK
# KOMENTARZ: Znaczącej poprawy nie widać, wzrost przy 20 000 wynika z innych danych


### WYBÓR: 10 000, Laplace: 1, Freq: 20
# Wynik: 44.2 + 41.0 = 85.2 %    OK


#====== Drzewa decyzyjne ======

### BEZ WZMACNIANIA - 78.92 %   OK

### Parametr trials dla 10 000 danych
# trials = 8  : 81.6 %    OK 
# trials = 10 : 80.6 %    OK
# trials = 12 : 81.7 %    OK
# trials = 13 : 82.2 %    OK    
# trials = 14 : 82.3 %    OK
# trials = 15 : 82.1 %    OK 
# KOMENTARZ: widać, że dokładność się zwiększa wraz z parametrem trials

# 16-20
accuracy_bez_wczytywania_modelu_drzewa = c(0.8272, 0.8216, 0.8276, 0.8284, 0.8284)


## Model do testowania
library(C50)

drzewo <- C5.0(imdb_train, imdb_train_labels)
#summary(drzewo)
predykcja = predict(drzewo, imdb_test)

confusionMatrix(predykcja, imdb_test_labels)$overall['Accuracy']


### sprawdzam model dla wartości trials 16:20
drzewa_model <- list()
pred_drzewa <- list()
cross_table_drzewa <- list()
accuracy <- NULL

for (i in 16:20){
drzewa_model[[i-15]] <- C5.0(imdb_train, imdb_train_labels, trials = i)

pred_drzewa[[i-15]] = predict(drzewa_model[[i-15]], imdb_test)

accuracy[i-15] = confusionMatrix(pred_drzewa[[i-15]], imdb_test_labels)$overall['Accuracy']

}

accuracy
accuracy_bez_wczytywania_modelu_drzewa = c(0.8272, 0.8216, 0.8276, 0.8284, 0.8284)

# widać, że wyniki nie sięgaja wartości powyżej 83 %

## "najlepszy"(to ze na teście wyszło najlepiej nie znaczy ze akurat ten parametr najlepszy) 
##  najlepszym wyborem wydaje się być wybranie trials w pograniczu 14

cross_table_drzewa = CrossTable(pred_drzewa[[2]], imdb_test_labels,
 prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
 dnn = c('predicted', 'actual'))

confusion_plot(cross_table_drzewa)

## ramka danych z wynikami dla innych parametrów trials 16:20
trials_scores = data.frame(trials = seq(16,20), accuracy = accuracy_bez_wczytywania_modelu_drzewa)