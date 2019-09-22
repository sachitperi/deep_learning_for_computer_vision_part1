A good application of checkpointing is to serialize your network to disk each time there is an 
improving during training. We define an improvement to be either a decrease in los or an increase 
in accuracy. The parameter can be set inside the keras callback. 

#### Checkpointing the Best Neural Network only 
Biggest downside of check pointing the is that we end up with a bunch of extra files that we are 
unlikely interested in. In this case it is best to save only one model and simply overwrite it 
every time with our metric improves during training. 
