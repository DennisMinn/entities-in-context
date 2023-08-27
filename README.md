This project examines the resilience of LLMs against entity replacement by analyzing data from three datasets: bAbI (four tasks), CLUTRR, and GSM8k.

There are two methodologies for facilitating named entity replacements in datasets and generating subsequent answers:

**Approach 1:**
In this method, the model reads the raw dataset, performs the named entity replacement directly, and then parses the data for further utilization. One primary advantage of this approach is its minimal storage requirement. However, its complexity can lead to confusion among readers and can challenge the reproducibility of results. Therefore, we introduce a more straightforward second approach, available in the "Approach_2" folder, designed to streamline the replication of results.

**Approach 2:**
Here, named entity replacements are performed directly on the datasets, and the augmented datasets are saved as .json files. These .json files are then input into LLMs to produce answers. It's worth noting that a major limitation of this method is its extensive storage requirement. Due to GitHub's storage constraints, we have uploaded the augmented dataset only for the initial version of bootstrap names. The datasets for the second and third sets of bootstrap names can be effortlessly created by substituting the new names into the .json files.

The code used to generate the plots showcased in this paper is located at "Approach_2/results/Analyze_results.ipynb".