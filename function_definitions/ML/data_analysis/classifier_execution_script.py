import pandas as pd
import re
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import wordnet
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.datasets import load_iris 
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
import pickle	
from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from function_definitions.ML.data_preparation.filtering_sms_script import filtered_df_func
from scipy import sparse, io
import sys





#account_type_dict={'CASA':1,'Credit_Card':2,'Debit_Card':3,'Loan':4,'Prepaid_Card':5,'Wallet':6,'_NA_':7}


def text_classify(column_dict,column_name,train_df):
	
	"""
	filtering out noise data from the training data
	"""
	train_df=filtered_df_func(train_df)
	
	print (train_df.shape)
	#test_df=test_df[['Message']]
	"""
	creating dictionary for representing 'Output_values' in Numerical format 
	"""
	#global column_dict
	#global account_type_dict
	msg_list=[]
	label_list=[]
	
	"""
	appending numbers for output labels eg:Acknowledgement':0,'Advert':1,
	"""
	#-------------------------for message_type
	for idx,row in train_df.iterrows():
		
		train_df.at[idx,column_name+'_id']=column_dict[train_df.at[idx,column_name]]
		msg_list.append(str(train_df.at[idx,'filtered_msg']))
		label_list.append(int(train_df.at[idx,column_name+'_id']))
	

	#-------------------------for account type
	#for idx,row in train_df.iterrows():
		#print train_df.at[idx,'AccountType'],'pppppppp' 
		
		#train_df.at[idx,'Account_Type_id']=account_type_dict[train_df.at[idx,'AccountType']]
		#msg_list.append(str(train_df.at[idx,'filtered_msg']))
		#label_list.append(int(train_df.at[idx,'Account_Type_id']))
	
	
	
	
	#train_df.to_csv('feature_sample.csv')
	print (train_df.head())
	#print msg_list
	
	#-------------------------for account type
	#for idx,row in test_df.iterrows():
		#test_df.at[idx,'Account_Type_id']=account_type_dict[test_df.at[idx,'AccountType']]
		#test_label_list.append(int(test_df.at[idx,'Account_Type_id']))
	
	#train_df.to_csv('feature_sample.csv')
	#print (train_df.head())
	#print msg_list
	
	
	"""
	initialization of vectorizer
	Using bi-grams so as to not lose context of the data
	"""
	vect=CountVectorizer(ngram_range=(1, 2),stop_words='english')
	#vect=HashingVectorizer(ngram_range=(1, 2))
	#tfidf_transformer = TfidfTransformer()
	#vect = TfidfVectorizer(use_idf='False',ngram_range=(1, 2))
	#vect = TfidfVectorizer(stop_words='english')
	"""
	Method to convert the text data and fit in matrix form
	"""
	X_train_dtm=vect.fit_transform(msg_list)
	#print X_train_dtm,type(X_train_dtm),'zzzzzzzzzzzzzzzzzz'
	save_vectorizer = open("vocabulary/"+column_name+"_vectorizer.pickle","wb")
	pickle.dump(vect, save_vectorizer)
	save_vectorizer.close()
	
	#tmp=pd.DataFrame(X_train_dtm.toarray(),columns=vect.get_feature_names())
	#tmp.to_csv('tfidf.csv')
	#raw_input()
	
	#X_new_counts = vect.transform(msg_list)
	#X_train_tfidf=tfidf_transformer.fit_transform(X_train_dtm)
	#print X_train_dtm
	#print vect.get_feature_names()
	
	
	#tmp=pd.DataFrame(X_train_dtm.toarray(),columns=vect.get_feature_names())
	#print tmp
	#tmp.to_csv('DTM.csv')
	#raw_input()
	
	#sgd=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)	
	rfc = RandomForestClassifier(n_estimators=100,n_jobs=1, max_depth=None,min_samples_split=2, random_state=0)
	#sgd=SGDClassifier()
	#nb=MultinomialNB()
	#clf = svm.SVC()
	#gnb = GaussianNB()
	
	
	
	#------------feature_selection algos--->chi2
	#X_new_chi2 = SelectKBest(chi2, k=2).fit_transform(X_train_dtm,label_list)
	
	
	
	
	print ('in',len(msg_list),len(label_list))
	"""
	Training the data using RFC classifier
	"""
	rfc.fit(X_train_dtm,label_list)
	importance = rfc.feature_importances_
	
	
	"""
	creating pickle file
	"""
	
	save_classifier = open("pickle/"+column_name+"_test_classifer.pickle","wb")
	pickle.dump(rfc, save_classifier)
	save_classifier.close()
	
	
	
	
	#print X_train_dtm
	#print importance,len(importance)
	#raw_input()
	#f_names_df=pd.DataFrame(X_train_dtm.toarray(),columns=vect.get_feature_names())
	#print f_names_df
	
	#sgd.fit(X_new_chi2,label_list)
	#sgd.fit(X_train_tfidf,label_list)
	#clf.fit(X_train_tfidf,label_list)
	#nb.fit(X_train_dtm,label_list)
	#gnb.fit(X_train_dtm.toarray(),label_list)
	
	print ('out')
	
	
	# dont fit just transfotrm the op data
	#tem=['You have successfully unlocked your SBI Card Online Account.','zzzzzz','Available bal. as on 13\/01\/16 05:15:28 PM IST   A\/c xxxx1767: INR 50.57Canara Bank','Your a\/c no. XXXXXXXX9201 is credited by Rs.5,000.00 on 07-04-16 by a\/c linked to mobile 9XXXXXX552(IMPS Ref no 609812347647)','Your account no XXXXXXXXX1767 is debited with Rs 237.06 on 23\/01\/2016 Towards POS withdrawal . Available balance is Rs 575.51','OTP to change login password is:78474403. Do not share it with anyone','You have received a bill of Rs. 442.05 for Vodafone Mumbai on 02-Mar-2016 . Pay via Net\/Mobile Banking\/Customer Contact Centre\/Branch.']
	#tem=['aaaaaaaa','zzzzzzzzz','lllllllll','ppppppppp']
	#f_msg=filtering_msg_func(tem)
	#print f_msg,'fffffffff'
	
	"""
	transforming the test data
	"""
def predict_test_data(column_dict,column_name,test_df):
	

	test_label_list=[]
	#vect=CountVectorizer(ngram_range=(1, 2))
	
	"""
	loading the trained vocabulary from the trained data pickle.
	"""
	
	test_list=test_df['Message'].tolist()
	#test_list=tem
	try:
		#print '1'
		vectorizer_f = open("vocabulary/"+column_name+"_vectorizer.pickle","rb")
		#print '2'
		vect=pickle.load(vectorizer_f)
		#print '3'
		vectorizer_f.close()
		# print '4'
		X_test_dtm=vect.transform(test_df['Message'].values.astype('U'))
	except Exception as e:
		print 'error in loading vectorizer:'+str(e)

	
		
	"""
	generating output IDS for test_data o use it for checking the accuracy
	"""
	
	#-------------------------for message type
	try:
		for idx,row in test_df.iterrows():
			test_df.at[idx,column_name+'_id']=column_dict[test_df.at[idx,column_name]]
			test_label_list.append(int(test_df.at[idx,column_name+'_id']))
	except Exception as e:
		print 'error in appending colums'+str(e)
	
	
	
	#X_test_tfidf=tfidf_transformer.transform(X_test_dtm)
	#print pd.DataFrame(X_test_dtm.toarray(),columns=vect.get_feature_names())
	#raw_input()
	
	"""
	loading the pickle file which contains the classifier (when used in production environement)
	"""
	t_start = time()*1000
	print ('innnn')
	classifier_f = open("pickle/"+column_name+"_test_classifer.pickle", "rb")
	classifier = pickle.load(classifier_f)
	classifier_f.close()
	
	"""
	Predicting the final output
	"""
	#print 'oyeeeee'
	#raw_input()
	#print X_test_dtm
	y_pred_class=classifier.predict(X_test_dtm)
	#probabilities = classifier.predict_proba(X_test_dtm)
	
	#print (probabilities,type(probabilities),'zzzz')
	#input()

	
	
	#y_pred_class=sgd.predict(X_test_tfidf)
	#y_pred_class=nb.predict(X_test_dtm)
	#y_pred_class=nb.predict(X_test_tfidf)
	#y_pred_class=gnb.predict(X_test_dtm.toarray())
	t_end = time()*1000
	total_time=t_end-t_start
	print (total_time,'timmmmmmmmmmmeeeeeeeeee')
	
	
	print (y_pred_class,type(y_pred_class),'zzzz')
	
	
	
	#--------------------for message type
	"""
	c=0
	for idx,row in test_df.iterrows():
		
		
		#for getting the key from value
		key = list(column_dict)[column_dict.values().index(y_pred_class[c])]
		#key = list(column_dict.keys())[list(column_dict.values()).index(y_pred_class[c])]
		#print key,y_pred_class[c],'yyyyy',c,'   ',y_pred_class
		#raw_input()
		test_df.at[idx,'ML_'+column_name]=key
			
		c=c+1
	"""
	#------Getting key  from values
	c=0
	for idx,row in test_df.iterrows():
		
		
		#for getting the key from value
		key = list(column_dict)[column_dict.values().index(y_pred_class[c])]
		#key = list(column_dict.keys())[list(column_dict.values()).index(y_pred_class[c])]
		#print key,y_pred_class[c],'yyyyy',c,'   ',y_pred_class
		#raw_input()
		test_df.at[idx,'ML_'+column_name]=key
			
		c=c+1
	
	
	
	print (test_df,'aaaaaaaa')
	
	#test_df=test_df[['Message',column_name,'filtered_msg',column_name+'_id','ML_'+column_name]]
	#raw_input()
	#test_df.to_csv("final_output_"+column_name+"_csv_new.csv")
	
	"""
	for calculating the accuracy by comparing train and test data
	"""
	print (np.mean(y_pred_class== test_label_list))
	#for accuracy detemination of a classifier
	cm=confusion_matrix(test_label_list,y_pred_class)
	print cm
	
	#feature_names = vect.get_feature_names()
	#print feature_names
	
	
	
	#print metrics.accuracy_score(label_list,y_pred_class)
		
	#[key for key in column_dict.items() if key[1] == value][0][0]
	test_df.to_csv(column_name+'_ML_final.csv')
	return test_df['ML_'+column_name].tolist()
	
		
	