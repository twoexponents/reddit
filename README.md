# Predicting Evolution of Conversations of Reddit

## Code Explanation
src/  
 -feature.all.py: Using all features  
 -feature.content/user/liwc/w2v.py: Using each feature only  
 -feature.separate.py: Let each feature have own RNN. (4 RNN -> fusion -> FC layer)  
 -feature.-.one.py: Using only the last comment's feature  
 -feature.-.generation.py: Showing the result of each epoch  
 -feature.-.case.py: Making list of sequences which are predicted well  
 -model.validation.py: Validate our RNN model with simple sequence and mnist  

extract/  
 : data extraction codes  
 -extract.convtree.py: Calculate each tree's tfidf and get top words of each sequence  
 -extract.sequences.innertree.py: separate training and test sequences by inner tree  
 -extract.sequences.outertree.py: separate training and test sequences by outer tree   

result/
 : no meaning yet  



## TODO
 -Make w2v features only with the top words
 -training set & test set tunning

