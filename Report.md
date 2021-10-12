# Which Physicochemicals Affect Red Wine Quality the Most?

Eyal Torn-Hibler

_eyal.torn-hibler@city.ac.uk_

 **Abstract** - The red wine industry is worth over $420 billion and as such, research into what makes a good tasting wine is important. Therefore, analysis has been conducted into which physicochemicals in Portuguese red wines have the most effects on the quality. Current research shows alcohol to be the most important chemical. This research would help consumers, sellers and wine producers in various ways. The analysis was conducted through visually exploring the data, as well as modelling machine learning models (AdaBoost, RandomForestEnsemble and DecisionTree) to output feature importance, then cross reference the findings. This research found alcohol, volatile acidity and sulphates to be the top three features that affect the quality of red wine. This has helped further confirm the current research that already exists on this topic.

**Keywords** — **red wine, analysis, data science.**

1. Introduction

The red wine industry was worth over $420 billion in 2020, and is expected to grow at a rate of 6.4%. Clearly, wine is a lucrative market and the importance of creating a good tasting wine is essential to having financial success in the industry. Therefore, it is important to understand what are the chemicals that make a high quality wine, how these can be produced, and which parameters in the wine production should be adjusted, all in the name of delicious wine. Most wineries cannot afford to conduct chemical analysis on their wines and many rely on intuition to produce a high quality wine. However, chemical analysis can help understand which chemicals influence the flavour of wine, assess if a batch of wine has been spoiled or even confirm if a wine has been adulterated or is fraudulent.

For example, understanding how compounds like sulphates, sugars, and density affect the quality of wine is important as it is possible for winemakers to add these into the wine to control the taste. This is especially crucial for very large wine producers whose customers expect a certain flavour profile from the wine. It is essential that they maintain this flavour profile to retain customers who enjoy its taste. Chemical analysis is also important with regards to food safety, where the levels of sugar and citric acid must be below a certain threshold.

Furthermore, companies who are selling wines can use information like this to confirm the authenticity of the label of the wine, as wine forgery becomes an increasingly large problem.

This research paper seeks to study which chemicals in wine most affect the quality. This will be done through visual analysis of the chemicals as well as using machine learning models to find the most important chemicals [1, 2].

2. Analytical Questions and Data

By the end of this paper it should be clear which chemical compounds in wine affect its flavour the most. I will answer this by investigating:

- How strongly correlated are variables in the dataset?
- Are there any features which are not directly correlated to quality but affect it?
- Does extracting feature importance from models help confirm which variables affect quality in the data visualization?

The dataset chosen has 1600 total data points and 11 different chemicals that are contained in Portuguese &#39;Vinho Verde&#39; wine (downloaded from Kaggle). One feature included is alcohol percentage. Knowing if this feature affects quality is useful for consumers when deciding which wine to buy, to ensure they choose the best tasting wine, as it is displayed on the label itself. These features are ones that winemakers will often measure and may even add to the wine. The dataset can be modelled into a classification problem, which means that by training and testing models on it, we can generate values for feature importance which can help confirm conclusions made when exploring the data. It is also assumed that all features have some influence on the quality of the wine, although this might be false.

The dataset has limitations, firstly only having 1600 rows makes this relatively small. Although this is large enough to generate strong analysis, more rows would strengthen the generated conclusions. Secondly, I categorised the dataset into &#39;good&#39; and &#39;bad&#39; qualities by labelling wines with 1&#39;s and 0&#39;s respectively. This assumes quality is an objective metric, however, it is known that there is a lot of noise and misclassification in wine data, which I will discuss further in my conclusion. The dataset also is limited in that the wines all come from a specific region, which again limits the generalizability of the conclusions generated.

3. Analysis

A brief overview of my analytical method: Firstly, the data was analysed visually to see which variables were correlated, and then three machine learning models were fitted to predict wine quality and subsequently used to produce feature importance. This gives two differing approaches in helping understand what features contribute to wine quality.

4. _Data Preparation_
The first step was to visualise the distribution of classes. This showed that the data was highly imbalanced with around 75% of the wines being scored between five and six. Therefore, to clearly show patterns in the data, better wines were grouped into &#39;good&#39; and &#39;bad&#39; quality which lead to the distribution being about 50% in each class. After removing duplicates, the dataset was now ready to visualize and saved for training the models laters.

Fig. 1 Bar graph showing distribution after duplicate removal
![Bar graph showing distribution after duplicate removal](eda\duplicates_per_label.png)

Having balanced data without duplicates is useful in identifying patterns as the plots will have the least possible amount of data to present information. This is useful as the clarity of the plots is increased without losing information.

5. _Data Visualization_

Fig. 2 Pearson&#39;s correlation matrix
![Fig. 2 Pearson's correlation matrix](eda\features_scatter_matrix.png)

The correlation matrix is the first key step in finding variables which affect wine quality. It measures the strength of a linear association between two variables. Firstly this shows that &#39;label&#39; or quality has a somewhat strong correlation to alcohol, sulphates and volatile acidity levels. Other correlations include &#39;citric acid&#39; with &#39;pH&#39;, &#39;fixed acidity&#39; with &#39;citric acid&#39;, &#39;fixed acidity&#39; with &#39;pH&#39; and &#39;total sulfur dioxide&#39; with &#39;free sulfur dioxide&#39;.

Plotting the Pearson&#39;s correlation is important as it serves as a guide upon which features to investigate further. I say &#39;only as a guide&#39; due to the assumption that the correlation has on the data, whereby the features plotted should follow a normal distribution. Therefore, more analysis should be conducted to confirm the results of the matrix as not all the data is exactly normally distributed [3, 4].

Fig. 3 Box plots of features correlated to quality
![Fig. 3 Box plots of features correlated to quality](eda\Features_correlation_quality.png)

These box plots help confirm the findings from the Pearson&#39;s matrix by showing that the features are roughly normally distributed (although with some outliers). Additionally, it shows the correlation between the features and that, indeed, as some increase/decrease, on average the quality of the wine improves.

Fig. 4 Scatter plots of features correlated with other features coloured by quality
![Fig. 4 Scatter plots of features correlated with other features coloured by quality](eda\Features_correlation_features.png)

Visualization on other correlated features is conducted to ensure there are no other underlying correlations that are unidentified by the Pearson&#39;s matrix. It is clear that there are no other hidden correlations.

Fig. 5 Regression plot
![Fig. 5 Regression plot](eda\Reg_plot.png)

Finally, a regression plot was conducted for each of the features shown in figure 4. However, this is the only one shown, as the others showed no difference in wine quality. This is interesting as we can see that wines that have an increased positive correlation between total sulfur, and free sulfur are on average of a lower quality. It would be interesting to know the average ratio of total sulfur to free sulfur for good and bad wines as there is some relationship here.

6. _Data Derivation_

Before modelling, the data was put through a transformation into a KNNImputer and a RobusterScaler which is useful due to the number of outliers in the data. This is also helpful to the models as some of them are sensitive to outliers and helped improve their accuracy.

7. _Construction of Models_

Three models were constructed; Decision Tree, Random Forest Ensemble and Adaptive Boosting. The models were then placed through a gridsearch to find the best parameters using 5 folds of cross validation.

 After many different hyperparameter combinations for the grid search, with varying results of overfitting, the final parameters chosen to conduct the grid search were narrowed down. Training had reached a point of diminishing returns in that the models&#39; accuracy was not improving, this was possibly due to outliers and noise in the data. These models were chosen to test different methods which can return feature importance.

8. _Validation of Results_

Fig. 6 Accuracy of models
![Fig. 6 Accuracy of models](models\model_accuracy.jpg)

The performance of the models was varied. Although, by using 3 models to make predictions this gives three tables of feature importance (as seen in the code). The ROC curves also help confirm the accuracy that the models are able to achieve.

All the models outputted &#39;alcohol&#39; and &#39;volatile acidity&#39; as the top two features that contribute to good or bad quality wine. The AdaBoost and Random Forest models both gave &#39;sulphates&#39; as the third most important feature, while the decision Tree gave &#39;residual sugar&#39;. As the performance of the decision tree was quite low (66.67%) the results it generated are not as strong. However the other two models performed well and interestingly, both gave &#39;total sulfur dioxide&#39; as the fourth most important feature. Having two well performing models generate the same results strengthens the conclusions that can be made from the outputs.

This helps cross reference findings in the data visualization and confirm at the top three features which most affect the quality of wine; &#39;alcohol&#39;, &#39;sulphates&#39; and &#39;volatile acidity&#39;. Interestingly&#39; citric acid&#39;, although visually showing to be correlated to quality, came lower down in feature importance across all models. Finally, &#39;total sulfur dioxide&#39; came fourth in two models for feature importance. When this is combined with the regression plot it confirms the correlation presented and the relationship it has with quality. It seems as though when the ratio between &#39;total sulfur dioxide&#39; and &#39;free sulfur dioxide&#39; becomes lower towards &#39;total sulfur dioxide&#39; the quality of the wine improves

9. Findings, Reflection and Further work

In conclusion, the study was successful in determining which features most affect the quality of wine. However, when comparing the visual analysis to the machine learning results, they only had 3 features in common. For example, citric acid had a correlation of 0.17, albeit a low positive score, and was ranked in feature importance, by the various models, near the bottom of their tables as one of the least important features. This inconsistency could be down to the performance of the models, as it becomes more difficult to rank less important features accurately.

Approaching the dataset from different points of analysis helped create a well rounded argument in deciding which of the variables were most important. However, a more data-centric approach could have helped strengthen the conclusions further.

The findings from this study will allow consumers of wine to pick better quality wine, producers of wine create better tasting wines and help sellers more accurately target wine buyers with the knowledge of a wine&#39;s physicochemicals.

10. _Limitations_

The largest limitation is that the dataset itself contains noise in that the quality of a wine is a completely arbitrary metric, as studies have shown people find it difficult to differentiate between expensive wines and cheap wines [5]. Therefore, whoever is classifying the wines into quality will have completely different results each time. This is also an explanation as to why it was so difficult to improve the models&#39; performance through grid searching.

11. _Further Work_

The study could have been improved by taking more steps to process the data and increase its quality. These include feature selection and removing outliers. It has been shown by Andrew Ng that fewer examples of clean data can improve model accuracy by the same amount as gathering twice the amount of noisy data [6]. This shows the importance and the impact that clean data can have on machine learning models. To further remove noise, more studies need to be done on wine tasting and preference, to find a more objective way of having people grade wine quality. Feature selection could have also been conducted to find chemicals which have very little impact on quality. Finally, a deeper chemical analysis could be conducted to cover all constituents in wine.

References

1. Wine Market Size, Share | Industry Report, 2021-2028 [WWW Document], n.d. URL[https://www.grandviewresearch.com/industry-analysis/wine-market](https://www.grandviewresearch.com/industry-analysis/wine-market) (accessed 8.23.21).
2. Christoph, N., Hermann, A., Wachter, H., 2015. 25 Years authentication of wine with stable isotope analysis in the European Union – Review and outlook. BIO Web of Conferences 5, 02020.[https://doi.org/10.1051/bioconf/20150502020](https://doi.org/10.1051/bioconf/20150502020)
3. Pearson Product-Moment Correlation - When you should run this test, the range of values the coefficient can take and how to measure strength of association. [WWW Document], n.d. URL[https://statistics.laerd.com/statistical-guides/pearson-correlation-coefficient-statistical-guide.php](https://statistics.laerd.com/statistical-guides/pearson-correlation-coefficient-statistical-guide.php) (accessed 8.24.21).
4. Schober, P., Boer, C., Schwarte, L.A., 2018. Correlation Coefficients: Appropriate Use and Interpretation. Anesthesia &amp; Analgesia 126, 1763–1768.[https://doi.org/10.1213/ANE.0000000000002864](https://doi.org/10.1213/ANE.0000000000002864)
5. The Wiseman &#39;Study&#39; – cheap versus expensive wine – Jamie Goode&#39;s wine blog, n.d. URL[https://www.wineanorak.com:/wineblog/wine-science/the-wiseman-%e2%80%98study%e2%80%99-%e2%80%93-cheap-versus-expensive-wine](about:blank) (accessed 8.24.21).
6. DeepLearningAI, n.d. A Chat with Andrew on MLOps: From Model-centric to Data-centric AI.
7. Red Wine Quality [WWW Document], n.d. URL[https://kaggle.com/uciml/red-wine-quality-cortez-et-al-2009](https://kaggle.com/uciml/red-wine-quality-cortez-et-al-2009) (accessed 8.24.21).

| Abstract | 127 |
| --- | --- |
| Introduction | 300 |
| Analytical Questions and Data | 300 |
| Analysis | 918 |
| Findings, Reflection and Further Work | 372 |