# Regressive_Parametric_Prediction_on_dry_reforming_reactions

Here using the concept of multi-feature parametric regressive prediction,on the hypothesis of Linear Regression, using gradient descent as the main backpropogation algorithm, we have developed a module to predict the Methane and Carbon Dioxide conversion, in a dry reform of a Biogas.


# Dataset 

The dataset is attached as the file named as dataset.csv. Here we have used a 24 feature dataset,keeping 70 percent as training dataset and 30 percent as the validation set.
The behavioural pattern of some of the input features is shown below.

![CO leaving reactor_vs_TimeRXNs at 800C1r csvaa](https://github.com/pranava1709/Regressive_Parametric_Prediction_on_dry_reforming_reactions/assets/60814171/120b9fca-cc42-4510-8b25-8984e21004bf)
![CO2 area_vs_TimeRXNs at 800C1r csvaa](https://github.com/pranava1709/Regressive_Parametric_Prediction_on_dry_reforming_reactions/assets/60814171/7057d35a-ab20-4ffd-8277-5b7343c1824d)
![total moles ENTERING GC_vs_TimeRXNs at 800C1r csvaa](https://github.com/pranava1709/Regressive_Parametric_Prediction_on_dry_reforming_reactions/assets/60814171/2f602a3d-4751-477c-a36c-0ba9338eecdf)
![total moles leaving the reactor_vs_TimeRXNs at 800C1r csvaa](https://github.com/pranava1709/Regressive_Parametric_Prediction_on_dry_reforming_reactions/assets/60814171/9f2e4581-1890-4643-aaf4-b423ab077f44)

# Results
We inferenced the model, on the initilaized valdiation set, and analyzed the performance based on the following equation:
score = max( 0 , 100*(1-mean_absolute_percentage_error(Actual,Predicted)))
Based on which, for Methane 99.51974844603062 % accurate, and for Carbon Dioxide 99.21633955636078 % accurate. 

![meth](https://github.com/pranava1709/Regressive_Parametric_Prediction_on_dry_reforming_reactions/assets/60814171/c54bb103-c1a6-422c-bf23-2022c6b107ad)
![carb](https://github.com/pranava1709/Regressive_Parametric_Prediction_on_dry_reforming_reactions/assets/60814171/961e7af1-ec00-428f-8c17-0d4dbc3b76a5)

Results, are based on the given dataset,which is particularly pretty small, will improve as the training samples would increase.

# Dependencies

Pandas 

Numpy

Sklearn

Matplotlib

OpenCV
