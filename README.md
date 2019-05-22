Predicting Evolution of Conversations of Reddit

src/
 -feature.all.py: Using all features

 -feature.content/user/liwc/w2v.py: Using each feature

 -feature.separate.py: Let each feature have own RNN. (4 RNN -> fusion -> FC layer)

 -model_validation.py: Validate RNN model with simple data

TODO(5/22): 
 -Solve the result of only 0(or 1) problem (caused by 'NaN' loss value)

