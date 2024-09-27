# NLP

1. **Importing the Dataset**:
   - The dataset containing restaurant reviews is imported. Each review is labeled as either positive (1) or negative (0).

2. **Text Preprocessing**:
   - Each review is cleaned by removing non-letter characters, converting to lowercase, and stemming words (e.g., "loved" becomes "love").
   - Stop words like "the", "is", "in" are removed to reduce noise, while "not" is kept to retain negation context.

3. **Creating Bag of Words Model**:
   - A **Bag of Words** model is created to convert text into a numerical format where each word is represented by its frequency in the review.
   - The most frequent 1500 words across all reviews are selected as features.

4. **Splitting Data into Training and Test Sets**:
   - The dataset is split into training (80%) and test (20%) sets, where the model is trained on the training data and tested on unseen data from the test set.

5. **Training the Naive Bayes Classifier**:
   - A **Naive Bayes classifier** is trained on the preprocessed reviews from the training set, learning to classify reviews as positive or negative.

6. **Predicting the Test Results**:
   - The model makes predictions on the test data, comparing the predicted sentiment with the actual sentiment to evaluate performance.

7. **Evaluating the Model**:
   - A **Confusion Matrix** is generated to assess the accuracy of the predictions, identifying how well the model classifies positive and negative reviews.
   - The accuracy score is computed to measure the overall effectiveness of the classifier.
