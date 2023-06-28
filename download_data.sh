mkdir data

# We took the Word Embedding and BiasBios data from: INLP(https://github.com/shauli-ravfogel/nullspace_projection), it was originally created by https://github.com/IBM/sensitive-subspace-robustness
# and performed the splits into train/dev/test same as INLP to 65/10/25 percent accordingly with a random split per profession.

# mkdir -p data/embeddings
# Check

# wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip -P data/embeddings/
# wget http://nlp.stanford.edu/data/glove.42B.300d.zip -P data/embeddings/
# unzip data/embeddings/crawl-300d-2M.vec.zip -d data/embeddings/                 
# unzip data/embeddings/glove.42B.300d.zip -d data/embeddings/ 

# mkdir -p data/biasbios
# Check

# wget https://storage.googleapis.com/ai2i/nullspace/biasbios/train.pickle -P data/biasbios/
# wget https://storage.googleapis.com/ai2i/nullspace/biasbios/dev.pickle -P data/biasbios/
# wget https://storage.googleapis.com/ai2i/nullspace/biasbios/test.pickle -P data/biasbios/


mkdir -p data/word-embedding
mkdir -p data/projection_matrix/word-embedding
# we will provide the saved datasets later
gdown google_file_ID -O data/word-embedding/

mkdir -p data/biography
mkdir -p data/projection_matrix/biography
# we will provide the saved datasets later
gdown google_file_ID -O data/biography/


