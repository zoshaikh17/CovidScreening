# CovidScreening

## Dataset
To assess the execution of our strategy on screening of COVID-19, we collected a multi-class multi-center chest Xray dataset. This dataset incorporates 2,196 illustrations with 
imagelevel names. Particularly, the dataset was collected from three different sources. The primary source is from a GitHub collection of chest X-rays of patients analyzed with 
COVID-19 . The second source is from a  Kaggle dataset2 ,  The collected information comprises of 196 chest X-rays with affirmed COVID-19, 1,000 chest X-rays with confirmed 
bacterial and viral pneumonia, and 1,000 examples of sound condition. We chosen out low-quality images in the dataset to anticipate superfluous classification errors.

https://github.com/ieee8023/covid-chestxray-dataset

https://www.kaggle.com/andrewmvd/convid19-x-rays

## Our Contributions
In summary, our main contributions are:
1) To improve the performance of the previously implemented models, we implemented an Oversampling method in Imbalanced Learning, called SMOTE on the COVID dataset.
2) We also tried to test the performance of MobileNet, GoogLeNet, DenseNet and CNN with and without using SMOTE. There was an improvement in the performance of all the models after performing SMOTE sampling on them.
3) We made our own CNN model.
4) Inorder to deal with the imbalanced dataset on DSCL and CNN models, we also tried to perform experiments using SMOTE-Variants.

