# German Inherent and Structural Case Representation 

This is a part of the code used in the paper Carslaw et al. (2026). If you use this code or datasets, please cite the following paper: 

Carslaw, I., Bárány, A., Kastner, I. & Steedman, M., (2026) “An LLM Investigation into Inherent and Structural Case Representation: a German Case Study”, Society for Computation in Linguistics 9(1). doi: https://doi.org/10.7275/scil.4161

Use is under a Creative Commons License.

---
## Packages 

In a conda environment, run the following code: 


```
conda install conda-forge::transformers
conda install conda-forge::pandas
conda install conda-forge::pytorch
conda install conda-forge::tqdm
conda install conda-forge::scikit-learn
conda install conda-forge::pypickle
conda install jmcmurray::os
conda install anaconda::regex
conda install anaconda::spacy-alignments
conda install anaconda::more-itertool
```

---
## Running Code 

Run the following command in the terminal - this code only runs the BERT and mBERT model:

```
python3 train_classifier.py model control dataset_type case_configuration caseless --case_spef CASE_SPEF
```

Command line arguments are case-sensitive.

```
model:                 (str) The model you want to test (mBERT or BERT). Input bert or mbert.
control:               (str) If you want to run a control probe (as detailed in appendix F in Carslaw et al., [2026]). To run the control probe, input True, otherwise input False. 
dataset_type:          (str) The specific control datasets you want to use (as detailed in sec 4.1 in Carslaw et al., [2026]) - the results presented in the paper are from the random dataset_type. Input random, noun_only, pronoun_only, or same_token_number.
case_configuration:    (int) The specific case configuration you want (as detailed in sec 3 in Carslaw et al., [2026]). Input 1 for {'acc':0, 'dat':1}, 2  {'nom':0, 'dat':1}, 3 for {'nom':0, 'acc':0, 'dat':1}, 4 for {'nom':1, 'acc':0, 'dat':1}, or 5 for {'nom':0, 'acc':1, 'dat':1}.  
caseless:              (str) If you want to run the cased experiment (experiment 1) or the caseless experiment (experiment 2). Input False for the cased experiment or True for the caseless experiment.
--case spef CASE_SPEF: (str) If control equals True, choose the specifc control probe you want (e.g. nom_probe, acc_probe, all_probe). Input nom, acc, or all. 
```  

The code will automatically save the results. For the experimental probes that is:

```
.\experiments\{dataset_type}\results\{model}\caseless_{caseless}\{case_configuration}.json
```

For the control probes the save path is: 

```
.\experiments\control\results\{model}\caseless_{caseless}\{CASE_SPEF}_random.json
#--case_spef all will be just saved as random.json
```


The code will automatically pickle and save the classifiers.

---
## Results 

The results described in the paper are in ```paper_results```. 

