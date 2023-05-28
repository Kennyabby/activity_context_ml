# activity_context_ml

# Overview

The ability to automatically recognize and comprehend human activities has grown in importance in today's fast-paced world across a variety of industries, from healthcare and sports analysis to smart homes and surveillance systems. Activity recognition, which is the process of identifying and categorizing human activities based on sensor data, is essential for enabling intelligent systems to comprehend and react to human behavior.

The goal of this project is to create an activity recognition model that accurately categorizes and forecasts various human activities. We can uncover patterns and connections hidden in sensor data by utilizing machine learning, which will allow us to build a powerful and effective activity recognition system.

In order to accomplish our goals, we will investigate and assess a variety of machine learning algorithms that have proven successful in activity recognition tasks. We will make use of publicly accessible datasets created specifically for activity recognition, which include information obtained from a variety of sensor sources, including magnetometers, accelerometers, and gyroscopes.

The project's phases will include data preprocessing, feature extraction, algorithm selection, model training, and performance assessment. Algorithms such as, Random Forests (RF), Multi-Layer Perceptron Neural Networks (MLPClassifier), Logistic Regression (LR) as well as more sophisticated methods like deep learning models, such as KNeighbors Classifier, will all be examined.

The development of a precise, effective, and well-capable activity recognition model with respect to generalizing to unknown data is crucial to the success of this project. By doing this, we can realize the potential for real-time, context-aware systems that can comprehend and modify their behavior in response to human activity.

# Problem Analysis

In many industries, like healthcare, sports analysis, and smart homes, activity recognition is essential. Real-time, context-aware systems that can comprehend and adapt to human behavior are made possible by the capacity to precisely and effectively distinguish and categorize human activities using sensor data. In order to better comprehend the difficulties and complexities involved in developing an activity recognition model using machine learning methods, we perform an extensive investigation of the issue at hand in this chapter. 

We begin by exploring the significance and relevance of activity recognition in different domains. An accurate activity identification model, for instance, can support early fall detection and remote patient monitoring in the healthcare industry. It can offer insightful information on athlete performance, technique, and injury avoidance in sports analysis. By adjusting to inhabitants' behaviors, activity recognition in smart homes can automate numerous operations and improve energy efficiency (Lara and Labrador, 2013).

We then go into the particular difficulties that activity recognition presents. Dealing with the inherent variety and complexity of human activities is one of the main challenges. The motion patterns, time span, level of intensity, and context of activities can all vary greatly. Accurately identifying activities is made more difficult by the noise, uncertainty, and sensor constraints that are present. Another difficulty is balancing the trade-off between accuracy and real-time speed, as the model must process data quickly without sacrificing recognition accuracy (Bulling et al., 2014).
Additionally, choosing and extracting useful information from sensor data is difficult. Careful thought must be put into selecting the most pertinent sensor modalities and extracting discriminative data from unprocessed sensor outputs. To extract significant patterns, feature engineering techniques including statistical measurements, must be investigated (Kwapisz et al., 2011).

For the purpose of building a powerful activity identification model, labeled training data must be readily available and of high quality. Large-scale datasets including various activities, environmental factors, and user populations can be difficult to collect and annotate since they need a lot of time and resources. The performance and generalizability of the model may also be impacted by dealing with class imbalance, where some activities may be underrepresented (Chen et al., 2018).

This analysis informs our subsequent steps in defining solution requirements and developing an effective activity recognition model using machine learning algorithms.

# Solution Requirements (Approach)

After examining the issue of activity recognition and the difficulties it presents, we now turn our attention to determining the criteria for a successful solution. The creation of our activity detection model using machine learning techniques should be guided by the important elements and factors we discover in this chapter.

Exploratory Data Analysis: We start by conducting a thorough Exploratory Data Analysis (EDA) on the dataset in order to ensure the construction of a reliable activity recognition solution. With the aid of pertinent Python libraries and functions, the EDA module will be put into use. The EDA involves the following tasks:

Loading and Exploring the Dataset: To load the dataset and fully investigate the data, we will develop code. This investigation will include looking for missing data points and using the proper cleaning procedures. The dataset will be subjected to descriptive statistical analysis, which will include measurements like mean, median, standard deviation, variance, minimum, maximum, skewness, and kurtosis. We will choose a variety of relevant variables and use tools like bar plots, grouped bar plots, and pie charts to examine their frequency and dependencies. We will analyze the results and present our conclusions through this analysis.

Class Distribution Analysis: We will look into the class distribution in the dataset during the EDA. To assess whether the classes are balanced or not, we shall map the distribution of the classes. If the classes are unbalanced, we shall deal with this problem by employing at least one useful strategy to balance the distribution of the classes. This phase is essential to ensuring that there is no bias in the activity recognition model toward any certain classes.

Dataset Splitting: Following data cleaning and analysis, the cleaned dataset will be divided into training and test datasets. In this step, the data is prepared for machine learning algorithm training and for performance testing of our activity recognition models.

Activity Classification: In this work, we focus on building activity classification models by training a range of machine learning methods on the dataset. 
One must first, in order to finish this task, do the following:

Model Training: At the very least, three classification models will be trained: KNeighbors Classifier, Random Forest Classifier, and Multi-Layer Perceptron Neural Networks. The features that were removed from the sensor data and the training dataset will be used to train these models. Each model will learn to classify actions based on the input labeled data.

Model Evaluation: Using the test dataset, we will assess the models' performance after they have been trained. Each model's accuracy, precision, recall, and F1-Score will be examined as part of the evaluation. To better understand how well the models perform in categorization, we will create a confusion matrix for each one. By comparing the results of the models, we can assess how well they perform in terms of correctly identifying and classifying various types of activities.


# Implementation of Solution (Results/Findings)

The technical details of an activity recognition solution employing machine learning methods are presented in this report. The implementation's objective was to create a reliable and accurate model that can categorize activities using sensor data. Exploratory data analysis, data preparation, data visualization, feature extraction and selection, model building and fitting, and performance evaluation were some of the major steps in the implementation process.

Exploratory Data Analysis (EDA): Utilizing the ed_analysis module, exploratory data analysis was the first step in the deployment. The raw dataset was loaded and checked for any errors or missing data points. To handle missing data and carry the appropriate data cleaning procedures, the clean_data function was used. In order to understand the features of the dataset, such as mean, median, standard deviation, variance, minimum, maximum, skewness, and kurtosis, the EDA module also contained descriptive statistical analysis. To ascertain whether the courses were balanced or not, the distribution of the classes was evaluated. If there was an imbalance, over sampling of the train data set was used to fix the problem with the class distribution (Figure 2.0) and (Figure 3.0). In order to prepare for model training, the dataset was then divided into training and test datasets.

Data Preprocessing: The data were prepared for model training during the preprocessing stage. The cleaned dataset was divided into features and the target variable using the DataFeaturizer class. To maintain uniformity and to remove the impact of various scales, the features were standardized. For the purpose of working with machine learning algorithms, the target variable was also encoded to numerical values (Table 1.0) and (Table 2.0). 

Data Visualization: Understanding the dataset and obtaining insights into the connections between the features and the goal variable were both made possible by data visualization. To see how each attribute and the target variable were distributed, histograms were used (Figure 1.0) and (Figure 2.0). To find any class imbalances, the distribution of the target variable was studied (Figure 1.0) and (Figure 2.0). In order to show the relationships between each attribute and the target variable and to give a general overview of the data patterns, scatter plots were made. The histogram analysis showed that some activity classes, such as class 12 (‘standing’) and 10 (‘walking’) had a larger frequency in the 'activity' column than others (see Figure 1.0),. Understanding the class distribution and any biases in the dataset required knowledge of this information. The correlations between specific features and the target variable were visually represented using scatter plots. However, more statistical research or domain knowledge may be needed to interpret these graphs. Data visualization was a first stage in the process of finding any anomalies, trends, or connections in the dataset. Making wise choices during feature extraction and model selection was aided by it. It was feasible to better comprehend the dataset's properties and spot any difficulties or chances for model training and evaluation by viewing the data.

Feature Extraction and Selection: Statistical feature extraction was carried out using the FeatureExtractor class to extract pertinent data from the dataset. The dataset's features were retrieved, including mean, median, variance, standard deviation, root mean square, zero-crossing, and sum of squares. Then, using measures like information gain or mutual information, the FeatureSelector class was used to choose the most informative features. The chosen features were kept for use in the training and assessment of the model in the future (Figure 9.0).

Model Creation and Fitting: To develop and train classification models, the classification_module was used. The processed features and the encoded target variable were inputs to the Model class in the module. KNeighbors Classifier, Random Forest Classifier, Multi-Layer Perceptron Neural Networks (MLP), and Logistic Regression were the models used in this implementation. To evaluate model performance during training, the training dataset was divided into training and validation sets. The models were sequentially trained using the training data and then optimized using methods like cross-validation or hyper-parameter tuning.

Performance Evaluation: The classification_module's PerformanceMetric class was used to assess how well the trained models performed. The class contrasted the ground truth labels in the test dataset with the expected activity labels from the models. The model's performance was measured using performance metrics like accuracy, precision, recall, and F1-score to determine how well it classified activities. To see how well the model classified each activity class, confusion matrices were created. The confusion matrices and performance indicators revealed each model's advantages and disadvantages. See (Figure 4.0, 5.0, 5.0, 7.0) and (Table 3.0 and Table 4.0)

Generalization and Evaluation: The right evaluation approaches were used to make sure the models could be generalized. To evaluate the models' performance on unknown data and address overfitting concerns, cross-validation or stratified sampling techniques can be used. To analyze model performance thoroughly, other evaluation metrics, such as area under the receiver operating characteristic curve (AUC-ROC), might be taken into account. To verify the model's effectiveness across various scenarios or datasets, additional tests or sensitivity assessments can be carried out.

Scalability and Efficiency: The effectiveness and scalability of the implementation were taken into account. Scalability may be improved when dataset size grows by streamlining the code and using effective methods. It is possible to investigate methods like parallel computing or distributed computing to speed up model training and prediction procedures. Techniques for model compression can also be used to optimize inference and decrease memory footprint.


APPENDIX:


Figure 1.0: Imbalanced dataset of the target class.


Figure 2.0: Imbalanced dataset of encoded target class.





Table 1.0: Original Context Recognition Dataset

_id	orX	orY	orZ	rX	rY	rZ	accX	accY	accZ	gX	gY	gZ	mX	mY	mZ	lux	soundLevel	activity
0	1	125	-17	2	0.070997	-0.131696	-0.877469	-0.038307	2.681510	8.65743	-0.041316	2.67655	8.64271	-31.2	-35.6	-37.6	5000	49.56	Sitting
1	2	126	-17	2	0.071486	-0.131480	-0.878024	-0.038307	2.681510	8.65743	-0.054196	2.67834	8.64654	-31.2	-36.0	-37.2	5000	53.38	Sitting
2	3	127	-17	2	0.071401	-0.131551	-0.878799	0.153229	2.681510	8.65743	-0.056867	2.68004	8.65088	-31.2	-36.0	-37.2	5000	53.38	Sitting
3	4	127	-17	2	0.071401	-0.131551	-0.878799	0.153229	2.681510	8.65743	-0.056867	2.68004	8.65088	-31.2	-36.0	-37.2	5000	49.53	Sitting
4	5	127	-17	2	0.070772	-0.131888	-0.879645	0.153229	2.681510	8.65743	-0.049128	2.68130	8.65458	-31.2	-35.6	-36.8	5000	49.53	Sitting
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
205515	205516	286	6	-2	-0.067944	-0.042220	0.617903	-0.383072	-0.804452	10.61110	-0.191598	-1.69443	10.55590	31.2	12.8	-35.6	5000	60.07	DescendingStairs
205516	205517	286	6	-2	-0.067944	-0.042220	0.617903	-0.383072	-0.804452	10.61110	-0.191598	-1.69443	10.55590	31.2	12.8	-35.6	5000	60.07	DescendingStairs
205517	205518	286	6	-2	-0.067944	-0.042220	0.617903	-0.383072	-0.804452	10.61110	-0.191598	-1.69443	10.55590	31.2	12.8	-35.6	5000	60.07	DescendingStairs
205518	205519	288	4	-2	-0.066261	-0.039767	0.615725	-0.383072	-1.149220	10.61110	-0.203887	-1.62111	10.47650	31.6	12.4	-35.6	5000	59.40	DescendingStairs
205519	205520	288	4	-2	-0.066261	-0.039767	0.615725	-0.383072	-1.149220	10.61110	-0.203887	-1.62111	10.47650	31.6	12.4	-35.6	5000	59.40	DescendingStairs
205520 rows × 19 columns


Table 2.0: Table head showing the encoded target variable.

orX	orY	orZ	rX	rY	rZ	accX	accY	accZ	gX	gY	gZ	mX	mY	mZ	lux	soundLevel	activity
0	0.024989	-0.003398	0.0004	0.000014	-0.000026	-0.000175	-0.000008	0.000536	0.001731	-0.000008	0.000535	0.001728	-0.006237	-0.007117	-0.007517	0.999556	0.009908	10
1	0.025188	-0.003398	0.0004	0.000014	-0.000026	-0.000176	-0.000008	0.000536	0.001731	-0.000011	0.000535	0.001729	-0.006237	-0.007197	-0.007437	0.999544	0.010671	10
2	0.025388	-0.003398	0.0004	0.000014	-0.000026	-0.000176	0.000031	0.000536	0.001731	-0.000011	0.000536	0.001729	-0.006237	-0.007197	-0.007437	0.999539	0.010671	10
3	0.025388	-0.003398	0.0004	0.000014	-0.000026	-0.000176	0.000031	0.000536	0.001731	-0.000011	0.000536	0.001729	-0.006237	-0.007197	-0.007437	0.999546	0.009902	10
4	0.025389	-0.003398	0.0004	0.000014	-0.000026	-0.000176	0.000031	0.000536	0.001731	-0.000010	0.000536	0.001730	-0.006237	-0.007117	-0.007357	0.999548	0.009902	10














Figure 3.0: Balanced Train Test Dataset


Figure 4.0: KNeighborClassifier Confusion Matrix


Figure 5.0: RondomForest Classifier Confusion Matrix



















Figure 6.0: LogisticRegression Confusion Matrix
















Figure 7.0: MLPClassifier Confusion Matrix






















Figure 8.0: Heat map showing the correlation between the columns of the dataset























Figure 9.0: Heat map showing the correlation of the extracted features and target variables



Table 3.0: Classification report of each model (Without Feature Extraction)

RandomForestClassifier

              precision    recall  f1-score   support

         0.0       0.73      0.99      0.84       349
         1.0       0.93      1.00      0.96       709
         2.0       0.80      0.99      0.89      1105
         3.0       0.79      1.00      0.88       154
         4.0       0.90      0.99      0.94       164
         5.0       0.78      0.98      0.87      1016
         6.0       0.99      1.00      0.99      5193
         7.0       0.56      1.00      0.72        22
         8.0       0.53      1.00      0.70       149
         9.0       0.75      1.00      0.86       244
        10.0       1.00      0.98      0.99     15207
        11.0       0.92      0.99      0.95      3958
        12.0       0.99      0.91      0.95     12834

    accuracy                           0.96     41104
   macro avg       0.82      0.99      0.89     41104
weighted avg       0.97      0.96      0.96     41104


KNeighborsClassifier

              precision    recall  f1-score   support

         0.0       0.29      0.95      0.44       349
         1.0       0.73      0.95      0.83       709
         2.0       0.72      0.92      0.80      1105
         3.0       0.30      1.00      0.46       154
         4.0       0.59      1.00      0.74       164
         5.0       0.55      0.89      0.68      1016
         6.0       0.98      0.98      0.98      5193
         7.0       0.32      1.00      0.48        22
         8.0       0.28      0.97      0.44       149
         9.0       0.46      0.96      0.63       244
        10.0       0.99      0.96      0.97     15207
        11.0       0.84      0.97      0.90      3958
        12.0       0.99      0.74      0.85     12834

    accuracy                           0.89     41104
   macro avg       0.62      0.94      0.71     41104
weighted avg       0.94      0.89      0.90     41104


LogisticRegression

              precision    recall  f1-score   support

         0.0       0.05      0.14      0.07       349
         1.0       0.14      0.63      0.23       709
         2.0       0.17      0.26      0.20      1105
         3.0       0.01      0.86      0.02       154
         4.0       0.02      0.92      0.03       164
         5.0       0.01      0.00      0.01      1016
         6.0       0.56      0.39      0.46      5193
         7.0       0.00      0.05      0.00        22
         8.0       0.03      0.77      0.05       149
         9.0       0.09      0.46      0.15       244
        10.0       0.00      0.00      0.00     15207
        11.0       0.58      0.01      0.02      3958
        12.0       0.02      0.00      0.00     12834

    accuracy                           0.08     41104
   macro avg       0.13      0.34      0.10     41104
weighted avg       0.14      0.08      0.07     41104


MLPClassifier

              precision    recall  f1-score   support

         0.0       0.04      0.44      0.08       349
         1.0       0.28      0.10      0.15       709
         2.0       0.17      0.40      0.24      1105
         3.0       0.05      0.82      0.09       154
         4.0       0.01      0.10      0.02       164
         5.0       0.47      0.16      0.24      1016
         6.0       0.77      0.82      0.80      5193
         7.0       0.02      1.00      0.05        22
         8.0       0.04      0.45      0.07       149
         9.0       0.12      0.93      0.21       244
        10.0       0.84      0.68      0.75     15207
        11.0       0.89      0.00      0.01      3958
        12.0       0.87      0.51      0.65     12834

    accuracy                           0.55     41104
   macro avg       0.35      0.49      0.26     41104
weighted avg       0.79      0.55      0.60     41104



Table 4.0: Classification report of the models (After Feature Extraction)



RandomForestClassifier

              precision    recall  f1-score   support

         0.0       0.14      0.72      0.24       349
         1.0       0.45      0.80      0.58       709
         2.0       0.43      0.68      0.52      1105
         3.0       0.24      0.96      0.38       154
         4.0       0.27      0.90      0.41       164
         5.0       0.26      0.64      0.37      1016
         6.0       0.92      0.91      0.91      5193
         7.0       0.12      0.64      0.20        22
         8.0       0.08      0.73      0.14       149
         9.0       0.13      0.72      0.21       244
        10.0       0.95      0.81      0.87     15207
        11.0       0.62      0.74      0.67      3958
        12.0       0.88      0.49      0.63     12834

    accuracy                           0.70     41104
   macro avg       0.42      0.75      0.47     41104
weighted avg       0.83      0.70      0.74     41104


KNeighborsClassifier

              precision    recall  f1-score   support

         0.0       0.09      0.68      0.16       349
         1.0       0.34      0.66      0.45       709
         2.0       0.38      0.54      0.45      1105
         3.0       0.11      0.92      0.20       154
         4.0       0.20      0.91      0.33       164
         5.0       0.21      0.54      0.30      1016
         6.0       0.90      0.89      0.90      5193
         7.0       0.06      0.68      0.11        22
         8.0       0.05      0.75      0.10       149
         9.0       0.06      0.62      0.10       244
        10.0       0.95      0.78      0.86     15207
        11.0       0.61      0.62      0.62      3958
        12.0       0.90      0.31      0.46     12834

    accuracy                           0.62     41104
   macro avg       0.38      0.69      0.39     41104
weighted avg       0.83      0.62      0.67     41104


LogisticRegression

              precision    recall  f1-score   support

         0.0       0.00      0.00      0.00       349
         1.0       0.25      0.54      0.34       709
         2.0       0.00      0.00      0.00      1105
         3.0       0.00      0.32      0.01       154
         4.0       0.01      0.76      0.03       164
         5.0       0.00      0.00      0.00      1016
         6.0       0.00      0.00      0.00      5193
         7.0       0.00      0.00      0.00        22
         8.0       0.00      0.00      0.00       149
         9.0       0.06      0.06      0.06       244
        10.0       0.44      0.18      0.26     15207
        11.0       0.01      0.00      0.00      3958
        12.0       0.55      0.57      0.56     12834

    accuracy                           0.26     41104
   macro avg       0.10      0.19      0.10     41104
weighted avg       0.34      0.26      0.28     41104


MLPClassifier

              precision    recall  f1-score   support

         0.0       0.01      0.15      0.03       349
         1.0       0.25      0.65      0.36       709
         2.0       0.13      0.06      0.08      1105
         3.0       0.01      0.75      0.02       154
         4.0       0.04      0.65      0.07       164
         5.0       0.00      0.00      0.00      1016
         6.0       0.53      0.60      0.57      5193
         7.0       0.00      0.00      0.00        22
         8.0       0.02      0.22      0.04       149
         9.0       0.02      0.40      0.04       244
        10.0       0.51      0.05      0.10     15207
        11.0       0.12      0.03      0.05      3958
        12.0       0.63      0.30      0.41     12834

    accuracy                           0.21     41104
   macro avg       0.17      0.30      0.13     41104
weighted avg       0.47      0.21      0.25     41104

