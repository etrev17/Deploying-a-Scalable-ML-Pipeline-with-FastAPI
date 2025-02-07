# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Type: Random Forest
The model was trained using supervised learning techniques, leveraging  Random Forest. It implements a machine learning pipeline ,evaluates performance using precision, recall, and F1 score, and providing functions for model inference and serialization.

## Intended Use
Primary users are soley connecected for school project.

## Training Data
Dataset used: Census.csv  which is an extraction was done by Barry Becker from the 1994 Census database.The set included 14 features and 48,842 instances. Data types included numeric, string, and boolean types. 

## Evaluation Data
Precision: How many of the model’s positive predictions are actually correct.
Recall: How well the model finds all the actual positive cases.
F1 Score: A balance between precision and recall, giving an overall performance score.

## Metrics
Precision: 0.7284 (72.84% of predicted positives are actual positives)
Recall: 0.6166 (61.66% of actual positives are correctly identified)
F1 Score: 0.6679 (Harmonic mean of precision and recall, indicating balanced performance). Performance was tested under different conditions, including dataset variations and feature importance analyses

## Ethical Considerations
Privacy and Security: Compliance with data protection regulations
## Caveats and Recommendations
A reccomendation to improve the model is to prioritize improving precision to ensure the model's predictions are correct more often. Another reccomendation is to expand on datasets used so the model can train better.
