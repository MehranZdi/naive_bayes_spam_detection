# Naive Bayes Spam Detection

This project implements a **spam detection system** using a **Naive Bayes Classifier**. The system identifies spam emails based on their content, represented as word counts of the 3000 most common words across all emails.

## Dataset
The dataset used for this project is from [Kaggle](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv).

## Requirements
To run this project, install the following dependencies:

```plaintext
pandas
scikit-learn
notebook
kagglehub
```

Install them using:
```bash
pip install -r requirements.txt
```

## Project Structure
```
naive_bayes_spam_detection/
├── naive_bayes_spam_detection.ipynb  # Jupyter Notebook for interactive demonstration
├── README.md                         # Project documentation
├── requirements.txt                  # Dependencies
```

## Implementation
The project is divided into the following steps:

1. **Load the Dataset**:
   - Read the dataset from the provided URL or local file.
   - Extract features (word counts) and labels (spam or not spam).

2. **Preprocess the Data**:
   - Split the dataset into training and testing sets (80%-20%).

3. **Train the Naive Bayes Classifier**:
   - Use the `MultinomialNB` model from `scikit-learn`.

4. **Evaluate the Model**:
   - Compute the confusion matrix, classification report, and F1 score.

5. **Custom Prediction**:
   - A function to test the model with new word count inputs.

## Key Metrics
- **Confusion Matrix**: Evaluates the classification performance.
- **Classification Report**: Includes precision, recall, and F1 score.
- **F1 Score**: The harmonic mean of precision and recall, emphasizing model performance on both classes.


### Using Jupyter Notebook:
Launch the notebook and run the cells sequentially to see the results interactively.

## License
This project is licensed under the MIT License.

---
### Author
[MehranZdi](https://github.com/MehranZdi)

Feel free to contribute or report issues in the repository!
