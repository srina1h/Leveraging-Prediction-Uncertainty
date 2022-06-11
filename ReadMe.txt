Enhancing Quality of Predictions Using Uncertainty Estimation Techniques

Group No - C3

Members:
Shivanirudh S G - 185001146
Srinath Srinivasan - 185001205
Sujay Sathya - 185001174

Guide:
Dr. T T Mirnalinee

a. URL/Source for Dataset: CMU Book summary dataset - https://www.cs.cmu.edu/~dbamman/booksummaries.html


b.	i) Software requirements:
		Python 3.8+
		Check requirements.txt for full list of python libraries involved.
   ii) Hardware requirements:
   		6 core processor
   		16GB of RAM
		NVIDIA P100 GPU for training
   
c. Detailed instructions to execute source code

	Download the 'booksummaries.zip' file from the URL provided, and extract the folder.
	
	Run the following command to install all libraries involved.
		pip install -r requirements.txt
		
	Run the following script to process the dataset into train, validation and test sets, using the following command.
		python3 data_preprocessing.py
		
	Run the cells on the ipython file 'source_code.ipynb', following the headings for loading the data, generating embeddings, creating models, 
	training on the dataset, and running inference on the whole dataset or a single sample.
	
	For running inference on a single sample of your choice, use the index number of test_dataset[x] and Y_test[x] in the function single_MC_pred/single_EN_pred/single_BNN_pred. 
	NOTE: range of x is from 0 to 1925, both inclusive.
	
