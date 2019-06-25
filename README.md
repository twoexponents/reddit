Predicting Evolution of Conversations of Reddit

src/  
 -feature.all.py: Using all features  
 -feature.content/user/liwc/w2v.py: Using each feature only
 -feature.separate.py: Let each feature have own RNN. (4 RNN -> fusion -> FC layer)
 -feature.-.one.py: Using only the last comment's feature
 -feature.-.generation.py: Showing the result of each epoch
 -feature.-.case.py: Making list of sequences which are predicted well
 -model.validation.py: Validate our RNN model with simple sequence and mnist  

TODO(5/22):  
 -Solve the result of only 0(or 1) problem (caused by 'NaN' loss value)  

