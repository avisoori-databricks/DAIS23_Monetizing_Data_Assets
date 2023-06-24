# Databricks notebook source
mob_permits = spark.sql("SELECT * FROM `john_snow_labs_mobile_food`.`mobile_food`.`mobile_food_facility_permit`;").toPandas()
display(mob_permits)

# COMMAND ----------

mob_permits.columns

# COMMAND ----------

relevant_fields = [ 'Facility_Type', 'Permit_Number',  'Block_Lot_Number', 'Block_Number', 'Lot_Number',  'Permit_Status', 'Food_Items', 'Is_Prior_Existing_Permit', 'Expiration_Date', 'Zip_Codes', 'Neighborhoods']

# COMMAND ----------

permits_refined=mob_permits[relevant_fields]
display(permits_refined)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM john_snow_labs_mobile_food.mobile_food.mobile_food_schedule

# COMMAND ----------

mob_schedule = spark.sql("SELECT * FROM john_snow_labs_mobile_food.mobile_food.mobile_food_schedule").toPandas()
display(mob_schedule)

# COMMAND ----------

mob_schedule.columns

# COMMAND ----------

relevant_fields = ['Day_Order', 'Starting_Day_of_the_Week',  'Permit_Number',  'Food_Items_Sold', 'Location_Id',  'Block_Number', 'Lot_Number', 'Is_Cold_Truck']

# COMMAND ----------

mob_schedulepd = mob_schedule[relevant_fields]
display(mob_schedulepd)

# COMMAND ----------

display(permits_refined)

# COMMAND ----------

len(set(mob_schedulepd.Food_Items_Sold.to_list()))

# COMMAND ----------

mob_schedule.shape

# COMMAND ----------

permits_refined.columns

# COMMAND ----------

import pandas as pd

# Perform the join on 'Permit_Number'
merged_df = mob_schedulepd.merge(permits_refined, on='Permit_Number', how='inner')

# Display the joined DataFrame
display(merged_df)


# COMMAND ----------

set(merged_df.Permit_Status.to_list())

# COMMAND ----------

#only consider 'APPROVED' and 'ISSUED' permits
# Create a boolean condition to filter rows
condition = (merged_df['Permit_Status'] == 'APPROVED') | (merged_df['Permit_Status'] == 'ISSUED')

# COMMAND ----------

# Apply the condition to the DataFrame to filter rows
filtered_df = merged_df[condition]
filtered_df = filtered_df[['Day_Order', 'Starting_Day_of_the_Week', 'Food_Items_Sold', 'Block_Number_x', 'Lot_Number_x', 'Is_Cold_Truck','Facility_Type',  'Food_Items']]

# Print the filtered DataFrame
display(filtered_df)

# COMMAND ----------

filtered_df.columns

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS DAIS23_data_sharing

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG DAIS23_data_sharing;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS DAIS23_ML_DB;
# MAGIC USE DAIS23_ML_DB;
# MAGIC

# COMMAND ----------

spark.createDataFrame(filtered_df).write.saveAsTable('food_truck_ml_table')

# COMMAND ----------

set(filtered_df['Food_Items'].to_list())

# COMMAND ----------

mappings = {
    'acai bowl': 'acai bowl',
    'coffee': 'coffee',
    'noodles': 'noodles',
    'ice cream': 'ice cream',
    'burrito' : 'latin',
    'taco':'latin',
    'peruvian':'latin',
    'kebab':'mediterranean',
    'gyro' : 'mediterranean',
    'pretzel':'pretzel',
    'hot dogs':'hot dogs',

}

# COMMAND ----------

def get_food_type(food_item):
    for keyword in mappings:
        if keyword in food_item.lower():
            return mappings[keyword]
    return 'other'


# COMMAND ----------

filtered_df['food_types'] = filtered_df['Food_Items'].apply(get_food_type)

# COMMAND ----------

display(filtered_df)

# COMMAND ----------


filtered_df.drop(['Food_Items_Sold','Food_Items','Starting_Day_of_the_Week'], axis=1, inplace=True)

# COMMAND ----------

display(filtered_df)

# COMMAND ----------

columns_to_encode = ['Block_Number_x', 'Lot_Number_x', 'Is_Cold_Truck','Facility_Type']
filtered_df_encoded = pd.get_dummies(filtered_df, columns=columns_to_encode)

# COMMAND ----------

spark.createDataFrame(filtered_df).write.saveAsTable('food_truck_ml_table_refined')

# COMMAND ----------

display(filtered_df_encoded)

# COMMAND ----------

new_column_names = [col.replace(' ', '_').replace(',', '').replace(';', '') for col in filtered_df_encoded.columns]
filtered_df_encoded.columns = new_column_names

# COMMAND ----------

spark.createDataFrame(filtered_df_encoded).write.saveAsTable('food_truck_features')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM food_truck_features;