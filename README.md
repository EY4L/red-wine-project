# Which Physicochemicals Affect Red Wine Quality the Most?

This project was part of the MSc Data Science course at City Of London University

### Table of Contents

- data: The dataset in its original form and the processed dataset
- eda: Images from the analysis
- models: .pkl Files of the trained models and .csv output of test results
- Report.md: Full written report 
- EDA.ipynb & TRAIN.ipynb: Jupyter Notebook outputs of the python files
- eda.py & train.py: Python code for analysis and ml

### Aims

Through data exploration and machine learning this project seeks answers to the following questions:
- How strongly correlated are variables in the dataset?
- Are there any features which are not directly correlated to quality but affect it?
- Does extracting feature importance from ML models help validate which variables seem to affect quality, as found in the data exploration?

### Findings

The study was successful in determining which features most affect the quality of wine. However, when comparing the visual analysis to the machine learning results, they only had 3 features in common. For example, citric acid had a correlation of 0.17, albeit a low positive score, and was ranked in feature importance, by the various models,  near the bottom of their tables as one of the least important features. This inconsistency could be down to the performance of the models, as it becomes more difficult to rank less important features accurately.

Approaching the dataset from different points of analysis helped create a well rounded argument in deciding which of the variables were most important. However, a more data-centric approach could have helped strengthen the conclusions further.

The findings from this study will allow consumers of wine to pick better quality wine, producers of wine create better tasting wines and help sellers more accurately target wine buyers with the knowledge of  a wine’s physicochemicals.



