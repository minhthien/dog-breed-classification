# dog-breed-classification

the dog_breed_classification_data_preparation.py contain the code to clean and prepare the data

the dog_breed_classification.ipynb contaion the models build on the preapared data

the dog_breed_prject_summary.docx contain the summary of the project



Building model details:

the model1 is base line model use 2 cnn layer with 128 filters size 3X3 with maxpooling and a dense layer with 256 node and an output layer with 120 node . This model is way overfit

Since model1 is way over fit model2 i tried to redue the number of filters, adding batchnormailzation and dropout.the model2 contain 2 cnn layer with 32 filter adn filter size is 3/3. 2 dense layer with 128 nodes and 64 nodes and an output layer of 120 nodes in between there is a bunch of BatchNormalization and Dropout.
