# Seattle Terry Stop Data Analysis

##### Author: Spencer Hadel
***
## Overview

Recent tensions in the United States have led to a mistrust of police forces across the country, particularly due to the increasing strength of movements such as Black Lives Matter and increased cultural attention to the racial and ethnic disparity in many facets of life. There is increasing focus on the scope of what police officer's are legally able to do, and whether they use this right fairly.

One such disparity has been observed in Terry Stops (also known as 'stop-and-frisks'), when a police officer uses can legally temporarily detain a person based on 'reasonable suspicion' that the person may be involved in criminal activity. The officer has the right to physically 'frisk' the subject, and take whatever action they feel is necessary properly handle the situation.

The newly elected mayor of Seattle campaigned on a platform of police reform, and has hired our agency to analyze, test, and interpret the current Seattle police department's Terry Stop data, so that their selected Chief of Police can make meaningful changes to the system as it stands.

***


## Data
This analysis utilizes about 52,000 data entries of Seattle Terry Stops ([from data.seattle.gov](https://data.seattle.gov/Public-Safety/Terry-Stops/28ny-9ts8)), in the file [Terry_Stops.csv](./data/Terry_Stops.csv). This data has been collected from 2015 until the present, and includes the following pieces of information for each entry:

> * `Subject Age Group`: Subject Age Group (10 year increments) as reported by the officer.
> * `Subject ID`: Key, generated daily, identifying unique subjects in the dataset using a character to character match of first name and last name. "Null" values indicate an "anonymous" or "unidentified" subject. Subjects of a Terry Stop are not required to present identification.
> * `GO / SC Num`: General Offense or Street Check number, relating the Terry Stop to the parent report. This field may have a one to many relationship in the data.
> * `Terry Stop ID`: Key identifying unique Terry Stop reports.
> * `Stop Resolution`: Resolution of the stop as reported by the officer.
> * `Weapon Type`: Type of weapon, if any, identified during a search or frisk of the subject. Indicates "None" if no weapons was found.
> * `Officer ID`: Key identifying unique officers in the dataset.
> * `Officer YOB`: Year of birth, as reported by the officer.
> * `Officer Gender`: Gender of the officer, as reported by the officer.
> * `Officer Race`: Race of the officer, as reported by the officer.
> * `Subject Perceived Race`: Perceived race of the subject, as reported by the officer.
> * `Subject Perceived Gender`: Perceived gender of the subject, as reported by the officer.
> * `Reported Date`: Date the report was filed in the Records Management System (RMS). Not necessarily the date the stop occurred but generally within 1 day.
> * `Reported Time`: Time the stop was reported in the Records Management System (RMS). Not the time the stop occurred but generally within 10 hours.
> * `Initial Call Type`: Initial classification of the call as assigned by 911.
> * `Final Call Type`: Final classification of the call as assigned by the primary officer closing the event.
> * `Call Type`: How the call was received by the communication center.
> * `Officer Squad`: Functional squad assignment (not budget) of the officer as reported by the Data Analytics Platform (DAP).
> * `Arrest Flag`: Indicator of whether a "physical arrest" was made, of the subject, during the Terry Stop. Does not necessarily reflect a report of an arrest in the Records Management System (RMS).
> * `Frisk Flag`: Indicator of whether a "frisk" was conducted, by the officer, of the subject, during the Terry Stop.
> * `Precinct`: Precinct of the address associated with the underlying Computer Aided Dispatch (CAD) event. Not necessarily where the Terry Stop occurred.
> * `Sector`: Sector of the address associated with the underlying Computer Aided Dispatch (CAD) event. Not necessarily where the Terry Stop occurred.
> * `Beat`: Beat of the address associated with the underlying Computer Aided Dispatch (CAD) event. Not necessarily where the Terry Stop occurred.

## Methods
This analysis cleans null values, and adjust the data in the original csv file to be easier to use ad interpret. Then we visualize a few of the relationships between different variables.

Next, the data is preprocessed, namely by creating dummy variables for the categorical data, which makes up the vast majority of it, before splitting the data into training and testing sets to run through classification algorithms. 
Using an iterative approach, the analysis visualizes the scores of these models. The scores observed on the classifiers are particularly the Accuracy, Recall, Precision, and F1 scores with an added visualization of the Confusion Matrices for each model, so that it is easy to see the relationship between True Positives, False Positives, True Negatives, and False Negatives in each attempt.
The Classifiers tested in this analysis are: 

Logistic Regression, K Nearest Neighbors, Decision Trees, Random Forest, and XG Boost


Each of these classifiers met varying levels of success, and the most successful subsequently had their hyperparameters tuned using GridsearchCV to find the best combination of parameters to increase F1 score, our primary metric.

After iteratively attempting to increase the scores for the models, the analysis investigates new outcomes with different feature sets and new target variables, each yielding different results. This was done using Machine Learning Pipelines in conjunction with GridSearchCV to streamline the selection process


## Results

The early stages of this analysis when visualizing the data before modeling, show that there is in fact some sort of identifiable racial disparity in the Terry Stop data. 

![](./img/1_1-graph_arrests_race.PNG)

![](./img/2_1-graph_legal_race.PNG)

It was also observed in initial analyis that there was a very clear disparity in Terry Stops based on the race of the Subjects:

![](./img/2_1-graph_total_pop.PNG)

![](./img/2_2_graph_terry_stops_by_race.PNG)

Once Modeling began, howver, the initial target variable, 'Physical_Arrest', was challenging for the classification algorithms to predict with much efficiency. Nonetheless, further steps were taken to improve these models, by attempting to tune their hyperparameters.

### Baseline Logistic Regression Model (after using SMOTE for synthetic data):
![](./img/3_2-baseline_logreg_smote.PNG)
![](./img/3_2-baseline_logreg_smote.PNG)

### Best Initial Model Performance - XG Boost:
![](./img/4_1-best_of_first_target_xgb_synth.PNG)
![](./img/4_2-xgb_synth_scores.PNG)

![](./img/5-initial_comparison.PNG)

Then, Machine Learning Pipelines were used to test different target variables from the data. These had more success than the previous, but still lacked results to make confident or impactful statements, for the time being.


### Best Params and Scores for "Arrested" Target:
![](./img/7_1-best_arrested_arrestedsmotecv.PNG)
![](./img/7_2-best_arrested_scores.PNG)

### Best Params and Scores for "Legal Action Taken" Target:
![](./img/8_1-best_legal_legalcv.PNG)
![](./img/8_2-best_legal_scores.PNG)

### Final Model Comparison
![](./img/10-final_best_scores.PNG)


## Conclusion
Unfortunately, the analysis did yield the results it needed in creating classifiers to accurately predict Terry Stop Outcomes. This is likely to be caused by several factors:

-The data is incredibly varied, and does not seem to be entirely consistent. There are multiple different potential features that seem to point to a subject's arrest, and they do not give the same results. More standardized and detailed data keeping practices from seattle.gov would certainly help future investigations yield more meaningful results.

-The current analysis was specifically using Classification algorithms for single, binary targets. The abundance of different types of data points to the possible need for multi-target Classifiers and deeper, more complex Machine Learning tasks.

-It is clear from the initial visualization of the data that there is something to be found here. Unfortunately, the current analysis shows that there will need to be more varied and in-depth attempts to create Machine Learning models that can help us successfully identify that problem.

-Future investigations could be potentially directed at the performance of individual officers within the Seattle Police Department, or perhaps aimed at individual regions of Seattle that face higher disparity in Terry Stops.


## More Information
The full analysis can be found in three Jupyter Notebooks. The [Data Cleaning nd Visualization](./nb_1-terry_data_cleaning_analysis.ipynb), [Data Modeling](./nb_2-terry_models.ipynb), and [Additional Target and Feature Selection](./nb_3-new_features_and_targets.ipynb) notebooks show each step of the process. Further business conclusions can be found in the [presentation](./terry_pres.pdf).

## Repository Structure

```
├── data
├── img
├── README.md
├── imports.py
├── nb_1-terry_data_cleaning_analysis.ipynb
├── nb_2-terry_models.ipynb
├── nb_3-new_features_and_targets.ipynb
├── terry_pres.pdf
```
