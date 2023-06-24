# Databricks notebook source
from datasets import load_dataset
import plotly.express as px
import pandas as pd
datasets = load_dataset('SetFit/yelp_review_full')


# COMMAND ----------

yelp_sample = datasets['train'].to_pandas()
display(yelp_sample)

# COMMAND ----------

yelp_sample = yelp_sample[yelp_sample['label'].isin([0, 4])].sample(n=1000)


# COMMAND ----------

import numpy as np
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.decomposition import PCA
import plotly.express as px

# COMMAND ----------

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')



# COMMAND ----------

def generate_embeddings(text):
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
    return embeddings



# COMMAND ----------

max_seq_length = 1024

# Truncate the text column to maximum sequence length
yelp_sample['truncated_text'] = yelp_sample['text'].apply(lambda text: text[:max_seq_length])



# COMMAND ----------

yelp_sample['Embeddings'] = yelp_sample['truncated_text'].apply(generate_embeddings)


# COMMAND ----------

display(yelp_sample)

# COMMAND ----------

yelp_sample['label'] = yelp_sample['label'].apply(lambda x: 'positive' if x >= 4 else 'negative')


# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# COMMAND ----------

embeddings = yelp_sample['Embeddings'].tolist()
pca = PCA(n_components=3)
pca_result = pca.fit_transform(embeddings)
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
pca_df['labels'] = yelp_sample.label.to_list()
pca_df['text'] = yelp_sample.text.to_list()


# COMMAND ----------

fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='labels', hover_data=['text'])
fig.update_traces(hovertemplate='Text: %{customdata[0]}')


# COMMAND ----------

display(fig)

# COMMAND ----------

embedding_df = spark.createDataFrame(yelp_sample)
embedding_df.count()

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG dais23_data_sharing;
# MAGIC USE DATABASE dais23_ml_db;
# MAGIC

# COMMAND ----------

embedding_df.write.saveAsTable('embedding_table')

# COMMAND ----------

