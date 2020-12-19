# StanceClassification

### Clustering Usage
```shell
cd scripts/
# use lockdown/all tweets, tf-idf/roberta vectors, and draw elbow figures with min and max k values
python clustering.py lockdown tfidf -d --min_k 30 --max_k 40                      
python clustering.py all roberta -d --min_k 1000 --max_k 1010                      

# use lockdown/all tweets, tf-idf/roberta vectors, set cluster numbers to 5/1000 and write output to file
python clustering.py lockdown roberta -c 5
python clustering.py all tfidf -c 2000

```