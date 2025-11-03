## ADDED Requirements
### Requirement: Baseline SMS Spam Detection
The system MUST classify SMS messages as spam or ham using TF-IDF features and a linear SVM.

#### Scenario: Achieve baseline accuracy
- **GIVEN** the dataset from `dataset/sms_spam.csv`
- **WHEN** the Phase 1 training pipeline runs end-to-end
- **THEN** the evaluation reports an accuracy of at least 0.95 on the 20% test split
- **AND** the confusion matrix and classification report are printed to the console

### Requirement: Persist Phase 1 Artifacts
The training workflow MUST persist the fitted TF-IDF vectorizer and trained SVM model.

#### Scenario: Save artifacts after training
- **GIVEN** the SVM baseline finishes training
- **WHEN** the pipeline completes
- **THEN** `artifacts/tfidf_vectorizer.pkl` and `artifacts/svm_model.pkl` exist on disk
- **AND** the vectorizer and model can be reloaded without error
