# Databricks notebook source
import base64
import pickle


# COMMAND ----------

catalog = 'dais23_data_sharing'
schema = 'dais23_ml_db'
table = 'encoded_model'
model_df = spark.sql("SELECT * FROM {}.{}.{}".format(catalog, schema, table))

# COMMAND ----------

display(model_df)

# COMMAND ----------

# Retrieve the Base64 encoded string from the DataFrame
base64_string = model_df.select('Encoded_Model').first()[0]

# COMMAND ----------


# Convert the Base64 encoded string back to model objects
pickle_data = base64.b64decode(base64_string)
model_objects = pickle.loads(pickle_data)

# COMMAND ----------

model_objects

# COMMAND ----------

[item[0] for item in model_objects]

# COMMAND ----------

model = pickle.loads(model_objects[2][1])
type(model)