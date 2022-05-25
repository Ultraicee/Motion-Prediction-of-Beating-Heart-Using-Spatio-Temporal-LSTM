# Motion-Prediction-of-Beating-Heart-Using-Spatio-Temporal-LSTM
These code for the paper "Motion Prediction of Beating Heart Using Spatio-Temporal LSTM", see the [paper](https://ieeexplore.ieee.org/document/9721087)

`cyclePredictWithRNN` works for predicting the result of next time based on trained network.

`LSTM_Prediction_Func` is used to input train and test data, setup hyperparameters of LSTM, record the error.

`run_PredictionTests` is the entrence of our task , similar to a main function. All function above are called by it.

You can see more details in our code. If you want to cite our data or code for your research, please cite our paper in format：
```
@ARTICLE{9721087,  
author={Zhang, Wanruo and Yao, Guan and Yang, Bo and Zheng, Wenfeng and Liu, Chao},  
journal={IEEE Signal Processing Letters},   
title={Motion Prediction of Beating Heart Using Spatio-Temporal LSTM},   
year={2022},  
volume={29},  
number={},  
pages={787-791},  
doi={10.1109/LSP.2022.3154317}}
```
