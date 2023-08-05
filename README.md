# Regressive_Parametric_Prediction_on_dry_reforming_reactions

Here using the concept of multi-feature parametric regressive prediction,on the hypothesis of Linear Regression, using gradient descent as the main backpropagation algorithm, we have developed a module to predict the Methane and Carbon Dioxide conversion, in a dry reform of a Biogas.


# Dataset 

The dataset is attached as the file named as dataset.csv. Here we have used a 24-feature dataset, keeping 70 percent as the training dataset and 30 percent as the validation set.
The behavioral pattern of some of the input features is shown below.

![Ch4 leaving reactor_vs_TimeRXNs at 800C1r csvaa](https://github.com/pranava1709/Regressive_Parametric_Prediction_on_dry_reforming_reactions/assets/60814171/ae5bc65f-e487-4733-912e-63402e8bd542)


![CO2 area_vs_TimeRXNs at 800C1r csvaa](https://github.com/pranava1709/Regressive_Parametric_Prediction_on_dry_reforming_reactions/assets/60814171/c90502bb-2dc9-4a67-8a0b-8f742425a3c6)

![total moles ENTERING GC_vs_TimeRXNs at 800C1r csvaa](https://github.com/pranava1709/Regressive_Parametric_Prediction_on_dry_reforming_reactions/assets/60814171/08484a2f-1681-4446-b8b1-39490a87f3db)

![total moles leaving the reactor_vs_TimeRXNs at 800C1r csvaa](https://github.com/pranava1709/Regressive_Parametric_Prediction_on_dry_reforming_reactions/assets/60814171/95b1a143-713f-4d11-9a42-c0beb814f84c)

# Results
We inferenced the model, on the initialized validation set, and analyzed the performance based on the following equation:
score = max( 0 , 100*(1-mean_absolute_percentage_error(Actual,Predicted)))
Based on this, for Methane 99.51974844603062 % accurate, and for Carbon Dioxide 99.21633955636078 % accurate. 

![meth](https://github.com/pranava1709/Regressive_Parametric_Prediction_on_dry_reforming_reactions/assets/60814171/d43ef480-ebad-4849-b715-5e8359d7e66d)

![carb](https://github.com/pranava1709/Regressive_Parametric_Prediction_on_dry_reforming_reactions/assets/60814171/5e1ae30c-4037-44e6-8d8a-f3969505f544)


# Dependencies

Pandas 

Numpy

Sklearn

Matplotlib

OpenCV
