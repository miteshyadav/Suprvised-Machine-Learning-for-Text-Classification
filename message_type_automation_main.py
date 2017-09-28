import pandas as pd
import re
import numpy as np
from time import time

from function_definitions.ML.data_preparation.filtering_sms_script import filtered_df_func
from function_definitions.ML.data_analysis.classifier_execution_script import text_classify
from function_definitions.ML.data_analysis.classifier_execution_script import predict_test_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split



def messege_type_automation_main():
	
	print 'in'
	df=pd.read_csv('Bank_Sms_Classified.csv')
	print 'out'
	"""
	Splitting the whole dataset into two sets-->80% training data and 20% test data
	"""

	train_df, test_df = train_test_split(df, test_size = 0.2)
	#print train_df
	
	
	
	
	time_start=time()
	col_dict=create_column_dict(df,'messagetype')
	"""
	function that builds the trained model
	"""
	text_classify(col_dict,'messagetype',train_df)
	
	"""
	function which uses the trained model for prediction 
	"""
	classified_list=predict_test_data(col_dict,'messagetype',test_df)
	time_end=time()
	print (time_end-time_start)/60
	
	#print classified_list
	
	#text_classify(train_df,test_df)


"""
function for label encoding the column names into unique values and corresponding numbers
"""	
def create_column_dict(df,column_name):
	col_list=df[column_name].unique().tolist()
	col_dict={}
	i=0
	for col in col_list:
	
		col_dict[col]=i
		i=i+1
		
	print col_dict
	return col_dict
	
	

messege_type_automation_main()