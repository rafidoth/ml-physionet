1. The purpose we use Phyionet is for comparsion to existing know research, hence we apply a similar data processing to produce for training data, data processing and scripts is in data_processing/  approaches reference: https://github.com/akaraspt/deepsleepnet
2. We download the data from https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/, only *PSG.edf and *Hypnogram.edf will be processed and use step 1. for processing, please notice SC4002E0-PSG.edf, SC4002EC-Hypnogram.edf SC4362F0-PSG.edf SC4362FC-Hypnogram.edf are broken source data, they can not be processed, others should be fine. After processing, put at least one data *.npz file into data\testing, data\training, data\vaildation respectively
3. For testing our codes and models, we put one subject npz file into data\testing, data\training and data\vaildation respectively. 
4. To control what models you want to run, you just make change train_EGG.py in MODEL_TYPE = 'RNN'  # TODO: Change this to 'MLP', 'CNN', or 'RNN' according to your task  
5. best_models_with_20_subjects have the models we ran for the paper and presentation with 12 training subjects, 4 vaildation subjects and 4 testing subjects
6. run following to the codes:
6.1 for creating processed data from raw data , we first need to put the data that download from 2. to data_processing\input. But for easy to test the codes, we have already put 3 sample subjects dataset there.
6.2 conda env create -f environment.yml
6.3 conda activate bd4hfinalproject
6.4 go to data_processing (cd data_processing)
6.5 python edf_parser.py 
6.6 the processed data: *.npz will be in data_processing/output folder.
6.7 for testing, we have put processed data into data\testing, data\training and data\vaildation (copy the *.npz from data_processing/output to data\testing, data\training and data\vaildation respectively (at least one file in each of the folder))
6.8 cd code  (go into code folder) (assuming you are in the root folder)
6.9 python train_EGG.py
6.10 the output results models will be under output\physionet, for example: output\physionet\MyRNN.pth
6.11 There are 3 output plots under code folers: accuracies_curves.png and losses_curves.png are for training and vaildation dataset, confusion_matrix.png is for testing dataset
6.12 we final best model is output\physionet\MyRNN.pth

All in all for quick test (all assuming in root folder) just do :

1. conda env create -f environment.yml
2. conda activate bd4hfinalproject
3. cd data_processing
4. python edf_parser.py 
5. cd .. (go back to root folder)
6. cd code
7. python train_EGG.py
8. check results plots in code folders and models output in output\physionet\MyRNN.pth

