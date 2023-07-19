%Data preparation
%Imports the csv file into MATLAB as a table
T = readtable("text_emotion_data_filtered.csv");

%tokenises all the tweets in the file
documents = tokenizedDocument(T.Content);

%Creates a bagofWords containing all the tokenised tweets
BagofWords = bagOfWords(documents);

%Removes stop words and words with fewer then 100 occurrences in bag
newbag = removeWords(BagofWords,stopWords);
Bag = removeInfrequentWords(newbag,100);

%builds the TFIDF matrix for the resulting bag
Matrix = tfidf(Bag);
Matrix = full(Matrix);

%Builds a corresponding label vector from the column of sentiments
labels = categorical(T.sentiment);

%Features and Labels

%Creates one feature and label matrix for training the data,
%Selects the first 6432 rows for training
trfeatures = Matrix(1:6432,:);
trlabel = labels(1:6432,:);

%Creates one feature and label matrix for testing the data,
%Selects the rest of the rows in the TF-IDF matrix.
tefeatures = Matrix(6432:8040,:);
telabel = labels(6432:8040,:);

%Model Training and Evaluation

%Discriminant analysis Training

%Trains the data using Discriminant Function
Discrimantmodel = fitcdiscr(trfeatures,trlabel);
%Uses the predict function to make predictions about the testing features
predictionsDM = predict(Discrimantmodel,tefeatures);

%Evaluation of Discriminant analysis model

%Calculates the number of correct predictions by gathering the sum of how
%many times the testing label matches the predicted set.
correct_predictionsDM = sum(telabel==predictionsDM);

%Accuracy is measured by comparing the predictions from your models to the
%test labels in the dataset.
AccuracyOfDM = correct_predictionsDM /size(telabel,1)*100;
%Creates a figure to represent the confusion chart diagram.
figure(1)
%Creates a chart showing the correct and incorrect predictions made by the
%model.
DiscrimantmodelCM = confusionchart(telabel,predictionsDM);
title(sprintf('Discriminant Analysis Model Accuracy=%.2f',AccuracyOfDM));

%Naive Bayes
%Follows the same comments as above but uses the Naive Bayes Model instead.
Naivemodel = fitcnb(trfeatures,trlabel);
predictionsNM = predict(Naivemodel,tefeatures);

%Evaluation of Naive Bayes model
correct_predictionsNM = sum(telabel==predictionsNM);
AccuracyOfNM = correct_predictionsNM /size(telabel,1)*100;

figure(2)
NaivemodelCM = confusionchart(telabel,predictionsNM);
title(sprintf('Naive Bayes Model Accuracy=%.2f',AccuracyOfNM));

%K-Nearest Neighbour
%Follows the same comments as above but uses the K-Nearest Neighbour Model instead.
Knnmodel = fitcknn(trfeatures,trlabel);
predictionsKNN = predict(Knnmodel,tefeatures);

%Evaluation of K-Nearest Neighbour model
correct_predictionsKnn = sum(telabel==predictionsKNN);
AccuracyOfKnn = correct_predictionsKnn /size(telabel,1)*100;

figure(3)
knnmodelCM = confusionchart(telabel,predictionsKNN);
title(sprintf('K-Nearest Neighbour Accuracy=%.2f',AccuracyOfKnn));

%Prints an matrix comparing the 3 correct predictions for each model.
correct_predictionsAll = ["KnnModel Predictions =",correct_predictionsKnn,"NmModel Predictions =",correct_predictionsNM,"DmModel Predictions =",correct_predictionsDM]
%Prints an matrix comparing the 3 correct predictions for each model.
correct_AccuracyAll = ["K-Nearest Neighbour =",AccuracyOfKnn,"Naive Bayes =",AccuracyOfNM,"Discriminant Analysis =",AccuracyOfDM]









