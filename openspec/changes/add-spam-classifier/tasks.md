## 1. Data Ingestion
- [x] 1.1 Create a Python script that downloads the dataset from https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv into `dataset/sms_spam.csv`.

## 2. Data Preprocessing
- [x] 2.1 Load `dataset/sms_spam.csv` with pandas and name the columns `label` and `message`.
- [x] 2.2 Clean the `message` text by lowercasing, removing punctuation, and removing common English stopwords.
- [x] 2.3 Vectorize the cleaned messages with TF-IDF and persist the fitted vectorizer (e.g., `tfidf_vectorizer.pkl`).

## 3. Model Training & Evaluation
- [x] 3.1 Load the TF-IDF features and labels, split them 80/20 into training and testing sets.
- [x] 3.2 Train a baseline SVM classifier on the training data.
- [x] 3.3 Evaluate on the test set: print accuracy, classification report, and confusion matrix.
- [x] 3.4 Save the trained SVM model artifact (e.g., `svm_model.pkl`).

## 4. Phase 2 – Logistic Regression (Planning)
- [ ] 4.1 Scaffold a logistic regression training script that reuses the TF-IDF artifacts.
- [ ] 4.2 Define preprocessing hooks to share normalization between SVM and logistic workflows.
- [ ] 4.3 Plan evaluation outputs comparing Logistic Regression against the SVM baseline.
- [ ] 4.4 Outline documentation updates describing Phase 2 configuration and findings.
