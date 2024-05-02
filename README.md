# NLP Predictive Model of Salary from LinkedIn Job Postings

## Dataset:
https://www.kaggle.com/datasets/arshkon/linkedin-job-postings - job_posting

## Overview:

Our goal for the final project is to create a salary range classification model for job listings on LinkedIn. A common challenge for job seekers on sites such as LinkedIn is determining the compensation for the role, as this information is often implicit and up to negotiation down the recruiting process. We aim to build a predictive model extracting information from job postings and output a potential range of salaries associated with the role. This model could provide better informed decision-making for job seekers and facilitate a transparent and efficient job market.


## Implementation Outline:
### 1. Data Preprocessing: data_preprocess.ipynb
✅ Convert all salary information into yearly compensation. Drop rows without salary information. (Krysten)

✅ Divide the salary information into discrete classes: 0-50k, 50k-100k, 100k-150k, 150k+. (Krysten)

✅ Preprocess the job description text data by tokenizing, lowercasing, removing punctuation, stopwords, and performing stemming or lemmatization. (Mansheel)

Processed descriptions: processed_description.csv 


### 2. TF-IDF Vectorization:
✅ Calculate the TF-IDF scores for each term in the job postings relative to the entire corpus. (Mansheel)
✅ Output TF-IDF vectors each row represents a job posting and each column represents a unique term in the corpus. (Mansheel)

### 3. MPNet Vectorization: word_embedding.py
✅ Using the pretrained Masked Permuted Pre-trained Network to perform word embedding and vectorize job descriptions. (Ethan, Oona)

### 3. Classification Model Experiments: tfidf_models.ipynb, mpnet_models.ipynb
✅ Split the dataset into training and testing sets (80-20). The training set will be used to train the classification model, while the testing set will be used to evaluate its performance. (Krysten)

- Train a logistic regression, SVM, RandomForest, AdaBoost, XGBoost, Deep Neural Network to  classify salary range using either the TF-IDF or MPNET vectors as features and the labeled salary bins as the target variable. (Krysten)
  
### 4. Model Evaluation:
- Evaluate the performance of the logistic regression model on the testing set using metrics such as accuracy, precision, recall, F1-score, AUC. (Krysten)

## Github commands:
- Clone this repository:
```
git clone https://github.com/krystennguyen/nlp_salary_predictor.git
```
- Add new files before commit:
```
git add --all
```
- Committing changes:
```
git commit -am "preprocess job description"
```
- Push local changes to remote repository:
```
git push origin main
```
- Fetch new changes from remote repository:
```
git fetch
```
- Merge new changes from remote repository:
```
git pull
```
