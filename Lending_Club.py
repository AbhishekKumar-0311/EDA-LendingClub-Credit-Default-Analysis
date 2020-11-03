#!/usr/bin/env python
# coding: utf-8

# # Lending Club Case Study

# #### Environment Setup

# In[1]:


# To get multiple outputs in the same cell

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


# Import the EDA required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import plotly.express as px

# Import the generic utility libraries

import os
import random
import datetime as datetime

#Importing the function
from pandas_profiling import ProfileReport


# In[3]:


# Set the required global options

# To display all the columns in dataframe
pd.set_option( "display.max_columns", None)
pd.set_option( "display.max_rows", None)

# Setting the display fromat
# pd.set_option('display.float_format', lambda x: '%.2f' % x)

#pd.reset_option('display.float_format')

sns.set(style='whitegrid')


# #### Reading the Loan data csv file

# In[4]:


# Read the raw csv file 'loan.csv' - containing the basic data of the loans
# encoding - The type of encoding format needs to be used for data reading

loan_df = pd.read_csv('loan.csv', low_memory=False)


# In[5]:


# Displaying random records from the dataframe

loan_df.sample(3)

# print(loan.head(3))
# print(loan.tail(3))


# #### The random observations are looked upon to understand variables and it's possible values.
# Post this, we will initiate data cleaning.

# # Data Cleaning

# #### Conclusion from the First Impression of data
# 1. Many variables with missing/null values.
#         Action - Calculate the missing % and would be dropped if it is too high.
# 2. Few variables have values which are prefixed or suffixed with 'months','%','>','<' symbols.
#         Action - Such variables need to be cleaned to remove consistent prefix or suffix.
# 3. Variables like 'desc', 'purpose' and 'title' contains redundant information.
#         Action - Except one, others can be dropped.
# 

# #### Get the precise Insights
# The first go-to step should be to basic info of dataframe and of each variable like:
# 
# For Numeric & Non-numeric variables:
# 
#     1. total records count
#     2. non-null records count
#     3. null records count and it's percentage(%)
# <br>
# For object(non-numeric) variables:
# 
#     4. unique records count and it's percentage(%)
#     5. top values
#     6. frequency
# <br>
# For Numeric variables:
# 
#     7. mean
#     8. median
#     9. mode
#     10. standard deviation
#     11. Quantiles
#     12. minimum & maximum values
# 

# Below is a *USER-DEFINED* function to get all the above mentioned basic info

# In[6]:


# Creating a User-Defined function to get the additional info on describe()

def df_stats(df,dt_flg):
    '''This function replicates the df.describe() with few additional features
        1. total_count
        2. null_count
        3. duplicate_with_null_count
        4. Renaming the existing columns
        Parameter:
        1. df - DataFrame name
        2. dt_flg - Flag to indicate the datatype of the columns in dataframe
            Possible values are:
                0 - All numeric
                1 - All non-numeric or object
                2 - Both Numeric and Object columns
    '''
    
    if dt_flg == 0:
        contents = df.describe().T.reset_index()
        contents.rename(columns={'index':'col_name','count':'non_null_count'}, inplace=True)
        contents['total_count'] = len(df.index)
        contents['non_null_count'] = contents['non_null_count'].astype('int')
        contents['null_count'] = contents['total_count'] - contents['non_null_count']
        contents['null%'] = np.round(contents['null_count']/contents['total_count']*100,2)
        columns = ['col_name','total_count','non_null_count','null_count','mean','std','min','25%','50%','75%','max']
        contents = contents[columns].infer_objects()
    elif dt_flg == 1:
        contents = df.describe().T.reset_index()
        contents.rename(columns={'index':'col_name','count':'non_null_count','unique':'unique_wo_null_count'}, inplace=True)
        contents['total_count'] = len(df.index)
        contents['non_null_count'] = contents['non_null_count'].astype('int')
        contents['unique_wo_null_count'] = contents['unique_wo_null_count'].astype('float')
        contents['null_count'] = contents['total_count'] - contents['non_null_count']
        contents['null%'] = np.round(contents['null_count']/contents['total_count']*100,2)
        contents['unique%'] = np.round(contents['unique_wo_null_count']/contents['non_null_count']*100,2)
        contents['duplicate_wo_null_count'] = contents['non_null_count'] - contents['unique_wo_null_count']
        columns = ['col_name','total_count','non_null_count','null_count','null%','unique_wo_null_count','unique%','duplicate_wo_null_count','top','freq']
        contents = contents[columns].infer_objects()
    elif dt_flg == 2:
        contents = df.describe(include='all').T.reset_index()
        contents.rename(columns={'index':'col_name','count':'non_null_count','unique':'unique_wo_null_count'}, inplace=True)
        contents['total_count'] = len(df.index)
        contents['non_null_count'] = contents['non_null_count'].astype('int')
        contents['unique_wo_null_count'] = contents['unique_wo_null_count'].astype('float')
        contents['null_count'] = contents['total_count'] - contents['non_null_count']
        contents['null%'] = np.round(contents['null_count']/contents['total_count']*100,2)
        contents['unique%'] = np.round(contents['unique_wo_null_count']/contents['non_null_count']*100,2)
        contents['duplicate_wo_null_count'] = contents['non_null_count'] - contents['unique_wo_null_count']
        columns = ['col_name','total_count','non_null_count','null_count','null%','unique_wo_null_count','unique%','duplicate_wo_null_count','mean','std','min','25%','50%','75%','max','top','freq']
        contents = contents[columns].infer_objects()
    return contents


# In[7]:


# Using the user-defined function - df_stats() to get the descriptive stats

loan_stats = df_stats(loan_df,2)
loan_stats


# In[8]:


#Generate the profiling report

#profile = ProfileReport(loan)
#profile = ProfileReport(loan.sample(1000), title = 'Loan - Pandas Profiling Report', explorative=True)
#profile = ProfileReport(loan_df, title = 'Loan - Pandas Profiling Report', minimal=True)

# save to file

#profile.to_file('loan_profile_report.html')

#profile.to_widgets()
#profile.to_notebook_iframe()


# #### Handling the missing data - 1

# In[9]:


# Calculate the % of missing/null values

miss_pct = round(loan_df.isna().sum() * 100/len(loan_df),2)
miss_pct[miss_pct > 0]


# In[10]:


# Filtering the variables with missing % greater than equal to 90

miss_gt_90 = miss_pct[miss_pct >= 90]

len(miss_gt_90)

#miss_gt_90


# There are 56 variables with minimum 90% of missing values.

# Creating a list of these 56 columns to drop them from the dataframe.

# In[11]:


miss_gt_90_cols = miss_gt_90.index
miss_gt_90_cols


# #### Dropping these 56 variables and storing the result in a new dataframe

# In[12]:


# A new dataframe 'loan' is created without the 56 variables

loan = loan_df.drop(miss_gt_90_cols, axis=1).copy(deep=True)


# In[13]:


loan.shape


# After removing these columns, 55 columns remain.
# <br> There are few more variables with missing values, which will be handled again at later point in time.

# In[14]:


loan.sample(3)


# #### On looking the data, we find that variable term should be a numerical variable and the unnecessary suffix should be removed

# In[15]:


# Looking the values of variable term

loan['term'].unique()


# In[16]:


# Cleaning variable term by removing 'months' suffix

loan['term'] = [x.strip().split(' ')[0] for x in loan.term]


# In[17]:


# Changing the datatype for term as it contains numerical values

loan['term'] = loan['term'].astype('int')


# In[18]:


# Re-Looking the values of variable term ater cleaning

loan['term'].unique()


# The variable term looks good and has 2 numerical values

# In[19]:


# Another way

#loan['term'] = loan['term'].str.strip().str.split(' ').str[0]


# #### On looking the data, we find that variable int_rate should be a numerical variable and the unnecessary suffix '%' should be removed

# In[20]:


# Cleaning variable int_rate by removing '%' suffix

loan['int_rate'] = [x.strip('%') for x in loan.int_rate]


# In[21]:


# Changing the datatype for int_rate as it contains numerical values

loan['int_rate'] = loan['int_rate'].astype('float')


# In[22]:


# Re-Looking the first 5 values of int_rate after cleaning

loan['int_rate'].unique()[:5]


# The variable int_rate looks good and has numerical values

# #### On looking the data, we find that variable emp_length should be a numerical variable and the unnecessary suffix 'years' should be removed

# In[23]:


# Looking the possible values of emp_length

loan['emp_length'].unique()


# In[24]:


# Standardising the emp_length as per the business : for < 1 year the emp_length should be 0, and 10 for 10+ years of experience

loan.loc[loan['emp_length'] == '< 1 year', 'emp_length'] = '0'


# In[25]:


# Cleaning variable emp_length by keeping only digits

loan['emp_length'] = loan['emp_length'].str.extract(r'(\d*)')


# In[26]:


# Changing the datatype for emp_length as it contains numerical values

loan['emp_length'] = loan['emp_length'].astype('float')


# In[27]:


# Re-Looking the values of emp_length after cleaning

loan['emp_length'].unique()


# The variable emp_length looks good and has numerical values

# #### On looking the data, we find that variable revol_util should be a numerical variable and the unnecessary suffix '%' should be removed

# In[28]:


# Looking the first 5 values of revol_util

loan['revol_util'].unique()[:5]


# In[29]:


# Cleaning variable revol_util by removing '%' suffix

loan['revol_util'] = loan['revol_util'].str.strip('%')


# In[30]:


# Changing the datatype for emp_length as it contains numerical values

loan['revol_util'] = loan['revol_util'].astype('float')


# In[31]:


# Looking the first 5 values of revol_util

loan['revol_util'].unique()[:5]


# The variable revol_util looks good and has numerical values

# #### Changing date indicator variables to datetime type

# In[32]:


def str_to_dt(dt_varlist):
    for var in dt_varlist:
        loan[var] = pd.to_datetime(loan[var], format='%b-%y')


# In[33]:


# List of date columns

dt_varlist = ['issue_d','earliest_cr_line','last_pymnt_d','last_credit_pull_d']

# Calling the str_to_dt function
str_to_dt(dt_varlist)


# In[34]:


loan[dt_varlist].info()


# In[35]:


loan.sample(3)


# The objective is to **identify predictor variables of default** so that at the time of loan application, those variables would help to approve or reject the application.
# 
# The **customer behaviour variables** i.e, those which are **generated after the loan is approved** are not available  at the point of application and thus should be removed from the analysis dataframe. Such variables **cannot be the predictor variables.**
# 
# Also there are few variables with which have either random, constant or fully unique values which would not contribute to the analysis.
# 
# Such variables should be removed.

# These variables are:
# 1. __id__ : is not available at the time of application. Also a random number given to the loan. Hence can be removed.
# 2. __member_id__ : is not available at the time of application. Also a random number given to the member. Hence can be removed.
# 3. __pymnt_plan__: has value for entire dataset as 'n'. Hence can be removed.
# 4. __url__: is a URL for LC page for corresponding to each memberid. Hence can be removed.
# 5. __zip_code__: first 3 digits of the 5 digit zip code are visible, also is redundant with addr_state. Hence can be removed.
# 6. __initial_list_status__: has value for entire dataset as 'f' out of all the possible values. Hence can be removed.
# 7. __policy_code__: has value for entire dataset as '1', indicating that all are publicly available. Hence can be removed.
# 8. __application_type__: has value for entire dataset as 'INDIVIDUAL', indicating that all are individual applications not joint. Hence can be removed.
# 9. __acc_now_delinq__: has value for entire dataset is '0', therefore can be removed.
# 10. __delinq_amnt__:  has value for entire dataset is '0', therefore can be removed.
# 11. __funded_amnt__: has value for this column as almost equivalent to loan_amnt. Hence removing to avoid Multicollinearity.
# 12. __funded_amnt_inv__: has value for this column as almost equivalent to funded_amnt. Hence removing to avoid Multicollinearity. Also this wont be available at the time of decision making of funding a loan.
# 13. __title__: has value for this column as almost equivalent to purpose. Hence removing to avoid redundancy.
# 14. __desc__: has value for this column as almost equivalent to purpose. Hence removing to avoid redundancy.
# 
# Some customer behaviour variables:
# 
# 15. __total_pymnt__ : is not available at the time of application.
# 16. __total_pymnt_inv__ : is not available at the time of application.
# 17. __total_rec_prncp__ : is not available at the time of application.
# 18. __total_rec_int__ : is not available at the time of application.
# 19. __total_rec_late_fee__ : is not available at the time of application.
# 20. __recoveries__ : is not available at the time of application.
# 21. __collection_recovery_fee__ : is not available at the time of application.
# 22. __last_pymnt_d__ : is not available at the time of application.
# 23. __last_pymnt_amnt__ : is not available at the time of application.
# 
# These 23 variables are dropped now.

# In[36]:


loan.shape


# In[37]:


loan.delinq_amnt.unique()


# delinq_amnt will be dropped since it has only 0.

# In[38]:


columns_to_drop = ['id','member_id','pymnt_plan','url','zip_code','initial_list_status','policy_code','application_type','acc_now_delinq','delinq_amnt','funded_amnt','funded_amnt_inv','title','desc','out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt']
loan = loan.drop(columns_to_drop,axis=1)


# In[39]:


loan.shape


# The dataframe has now 30 variables and 39717 records.

# #### Handling the missing variables - 2
# 
# Let's examine more for the missing values.

# In[40]:


miss_ser = round(loan.isna().sum() * 100/len(loan),2).sort_values(ascending=False)
miss_ser[miss_ser > 0]


# The variables emp_title, emp_length have 6.19% and 2.71% missing value. These columns have information about the borrower like their job title and their employment length in years, which are important attributes. We ll remove the rows with nan/blank values for these varaibles.

# In[41]:


loan = loan[~loan.emp_title.isnull()]
loan = loan[~loan.emp_length.isnull()]


# In[42]:


loan.shape


# In[43]:


miss_ser = round(loan.isna().sum() * 100/len(loan),2).sort_values(ascending=False)
miss_ser[miss_ser > 0]


# In[44]:


loan.pub_rec_bankruptcies.unique()


# In[45]:


loan.chargeoff_within_12_mths.unique()


# Can be dropped as it has either 0 or NaN

# In[46]:


loan.collections_12_mths_ex_med.unique()


# Can be dropped as it has either 0 or NaN

# In[47]:


loan.tax_liens.unique()


# Can be dropped as it has either 0 or NaN

# In[48]:


loan.drop(['chargeoff_within_12_mths','collections_12_mths_ex_med','tax_liens'], axis=1, inplace=True)


# In[49]:


loan.pub_rec_bankruptcies.unique()


# In[50]:


miss_ser = round(loan.isna().sum() * 100/len(loan),2).sort_values(ascending=False)
miss_ser[miss_ser > 0]


# In[51]:


loan.drop(['mths_since_last_delinq'], axis=1, inplace=True)


# In[52]:


# Removing the records with missing values in pub_rec_bankruptcies

loan[loan.pub_rec_bankruptcies.isna()].index
loan.drop(loan[loan.pub_rec_bankruptcies.isna()].index, inplace=True)

# Checking the shape of df
loan.shape

# Removing the records with missing values in revol_util
loan[loan.revol_util.isna()].index
loan.drop(loan[loan.revol_util.isna()].index, inplace=True)

# Checking the shape of df
loan.shape


# #### The objective is to identify predictor variables of default so that at the time of loan application, those variables would help to approve or reject the application.
# 
# The ones with 'Current' value are neither Fully-paid or Defaulted. So removing these records from loan dataframe

# In[53]:


loan.loan_status.unique()


# In[54]:


loan = loan.loc[loan['loan_status'] != 'Current']


# In[55]:


loan.shape


# In[56]:


loan.loan_status.unique()


# ---
# 
# # Data Analysis
# 
# ---

# In[57]:


loan_df = loan.copy()


# In[58]:


loan_df.sample(3)


# ### in_default_flg - a business driven metric is created for the target variable 'loan_status'
# 
# 0 - Not Default or Fully Paid <br>
# 1 - Defaulter

# In[59]:


loan_df['in_default_flg'] = [0 if x == 'Fully Paid' else 1 for x in loan_df['loan_status']]
loan_df.sample(3)


# In[60]:


loan_df.in_default_flg.unique()


# In[61]:


loan_df.in_default_flg.value_counts()


# In[62]:


df1 = loan_df.groupby('in_default_flg')['loan_amnt'].sum().to_frame().reset_index()
df1['loan_amnt_pct'] = (df1['loan_amnt']/loan_df.loan_amnt.sum()*100).round(2)
df1


# In[63]:


#loan_df.groupby('in_default_flg')['loan_amnt'].sum().plot.bar();


# In[64]:


# df2 = loan_df.groupby('in_default_flg').size()/len(loan_df)*100
# df3 = df2.to_frame().reset_index()
# df3.rename(columns= {df3.columns[1] : 'count_pct'}, inplace=True)
# df3['count_pct'] = df3['count_pct'].round(2)
# df3
# #df3['count'] = df3['count'].apply(lambda x: round(x, 2))


# The above way is commented as it creates only count_pct.<br>
# The below method creates count as well count_pct.

# In[65]:


df2 = loan_df.groupby('in_default_flg').size().to_frame().reset_index()
df2.rename(columns= {df2.columns[1] : 'count'}, inplace=True)
df2['count_pct'] = (df2['count']/len(loan_df)*100).round(2)
df2
#df3['count'] = df3['count'].apply(lambda x: round(x, 2))


# In[66]:



## Show labels in bar plots - copied from https://stackoverflow.com/a/48372659
def showLabels(ax, d=None):
    plt.margins(0.2, 0.2)
    rects = ax.patches
    i = 0
    locs, labels = plt.xticks() 
    counts = {}
    if not d is None:
        for key, value in d.items():
            counts[str(key)] = value

    # For each bar: Place a label
    for rect in rects:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        if d is None:
            label = "{:.1f}".format(y_value)
        else:
            try:
                label = "{:.1f}".format(y_value) + "\nof " + str(counts[str(labels[i].get_text())])
            except:
                label = "{:.1f}".format(y_value)
        
        i = i+1

        # Create annotation
        plt.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.


# In[67]:


# plt.figure(figsize=(5,7))
# ax = loan_df.groupby('in_default_flg').loan_amnt.count().plot.bar();
# showLabels(ax);
# ax.set_ylabel('Count', fontweight ='bold')
# ax.set_xlabel('In_Default_Flg', fontweight ='bold') 
# ax.grid(True);
# ax.set_title('Count by in_default_flg', fontsize = 19, fontweight ='bold');


# In[121]:


plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
ax = sns.countplot(x="loan_status", data=loan_df);
showLabels(ax);
#ax.set_xlabel('Display X-axis Label', fontweight ='bold')
#ax.set_ylabel('Display X-axis Label', fontweight ='bold') 
ax.grid(True);
ax.set_title('Count by loan_status', fontsize = 19, fontweight ='bold');


plt.subplot(1,2,2)
ax = sns.barplot(x="loan_status", y= 'count_pct', data=df2);
showLabels(ax);
#ax.set_xlabel('Display X-axis Label', fontweight ='bold')
#ax.set_ylabel('Display X-axis Label', fontweight ='bold') 
ax.grid(True);
ax.set_title('Count % by loan_status', fontsize = 19, fontweight ='bold');


# ---
# Approximately **14%** of loans in the dataset are **defaulted.**
# <br>Any variable that increases percentage of default to higher than 16.5% should be considered a business risk. 
# (16.5 is 18% higher than 13.97 - a large enough increase) 
# 
# ---

# In[69]:


# plt.figure(figsize=(5,7))
# ax = sns.barplot(x="in_default_flg", y= 'loan_amnt', data=loan_df, estimator=lambda x: len(x) / len(loan_df) * 100)
# showLabels(ax);
# #ax.set_xlabel('Display X-axis Label', fontweight ='bold')
# #ax.set_ylabel('Display X-axis Label', fontweight ='bold') 
# ax.grid(True);
# ax.set_title('Count % by in_default_flg', fontsize = 19, fontweight ='bold');


# # Univariate Analysis - Categorical Features

# In[70]:


# def default_rate_per_var(var, df = loan_df, sort_flg=True, head=0):
    
#     plt.subplot(1, 2, 1)
#     if head == 0:
#         ser = (loan_df[var].value_counts(normalize=True)*100)
#     else:
#         ser = (loan_df[var].value_counts(normalize=True).head(head)*100)
#     #ser
#     if sort_flg:
#         ser = ser.sort_index()
#     ax = ser.plot.bar(color=sns.color_palette("Paired", 10))
#     ax.set_ylabel('% count in data', fontsize=16)
#     ax.set_xlabel(var, fontsize=12)
#     showLabels(ax)
#     plt.subplot(1, 2, 2)
#     if head == 0:
#         ser = (loan_df.loc[loan_df['in_default_flg'] == 1][var].value_counts(normalize=True)*100)
#     else:
#         ser = (loan_df.loc[loan_df['in_default_flg'] == 1][var].value_counts(normalize=True).head(head)*100)
#     #ser
#     if sort_flg:
#         ser = ser.sort_index()
#     ax = ser.plot.bar(color=sns.color_palette("Paired", 10))
#     ax.set_ylabel('% of Defaulted loans', fontsize=16)
#     ax.set_xlabel(var, fontsize=12)
#     showLabels(ax)


# In[71]:


def default_rate_per_var(var, df = loan_df, sort_flg=True, head=0):
    
    plt.subplot(1, 2, 1)
    if head == 0:
        ser = (df[var].value_counts(normalize=True)*100)
    else:
        ser = (df[var].value_counts(normalize=True).head(head)*100)
    #ser
    if sort_flg:
        ser = ser.sort_index()
    ax = ser.plot.bar(color=sns.color_palette("Paired", 10))
    ax.set_ylabel('% count in data', fontsize=16)
    ax.set_xlabel(var, fontsize=12)
    showLabels(ax)
    plt.subplot(1, 2, 2)
    if head == 0:
        ser = (df.loc[df['in_default_flg'] == 1][var].value_counts(normalize=True)*100)
    else:
        ser = (df.loc[df['in_default_flg'] == 1][var].value_counts(normalize=True).head(head)*100)
    #ser
    if sort_flg:
        ser = ser.sort_index()
    ax = ser.plot.bar(color=sns.color_palette("Paired", 10))
    ax.set_ylabel('% of Defaulted loans', fontsize=16)
    ax.set_xlabel(var, fontsize=12)
    showLabels(ax)


# In[72]:


plt.figure(figsize=(12,7));
default_rate_per_var('term');


# #### Inference :
# 
# - The default rate increases drastically for 60 months term.
# - So, **term** is an **important** driving parameter.

# In[73]:


plt.figure(figsize=(12,7));
default_rate_per_var('grade');


# #### Inference :
# 
# - There is a significant increase in default rate for C,D,E,F,G grades.
# - So, **grade** is an **important** driving parameter.

# In[74]:


plt.figure(figsize=(18,12));
default_rate_per_var('sub_grade');


# #### Inference :
# 
# - There is again an increase in default rate for C,D,E,F,G sub_grades.
# - So, **sub_grade** is an **important** driving parameter.

# In[75]:


plt.figure(figsize=(12,7));
default_rate_per_var('purpose',sort_flg=False);


# #### Inference :
# 
# - There is an increase in default rate for debt_consolidation purpose.
# - So, **debt_consolidation purpose** is more prone to default.

# In[76]:


plt.figure(figsize=(12,7));
default_rate_per_var('verification_status');


# In[77]:


loan_df.groupby('in_default_flg')['verification_status'].value_counts()


# #### Inference :
# 
# - There is no clear increase in default rate for different verification_status.
# - So, we wont consider **verification_status**.

# In[78]:


plt.figure(figsize=(12,7));
default_rate_per_var('home_ownership',sort_flg=False);


# #### Inference :
# 
# - There is no clear increase in default rate for different home ownership status.
# - So, we wont consider **home_ownership**.

# In[79]:


plt.figure(figsize=(12,7));
default_rate_per_var('pub_rec_bankruptcies');


# In[80]:


loan_df.groupby('in_default_flg')['pub_rec_bankruptcies'].value_counts()


# #### Inference :
# 
# - There is no clear increase in default rate for different pub_rec_bankruptcies values.
# - So, we wont consider **pub_rec_bankruptcies**.

# #### A user-defined function for segmented univariate analysis

# In[81]:


def segment_uni(x_axis, xlabel_rotation=0):
    ax = sns.barplot(x = x_axis,  y= 'in_default_flg', hue = 'loan_status', order= sorted(list(set(loan_df[x_axis]))), data = loan_df, estimator=lambda x: len(x) / len(loan_df) * 100 )
    showLabels(ax)
    #plt.title(figure_title)
    plt.xticks(rotation = xlabel_rotation)
    #plt.xlabel(xlabel, labelpad = 15)
    #plt.ylabel(ylabel, labelpad = 10)
    
    #if legend_flag == True:
        #plt.legend(loc = legend)


# In[82]:


# plt.subplot(1, 2, 1)
# loan_df.groupby(['in_default_flg','term']).size().plot.bar()
# plt.subplot(1, 2, 2)
# loan_df.groupby('in_default_flg')['term'].value_counts().plot.bar()
plt.figure(figsize=(12,7));
segment_uni('term');


# In[83]:


plt.figure(figsize=(12,7));
segment_uni('grade');


# #### Inference :
# 
# - There is a high difference between the default and non-default rate for better grades like A,B and steeply decreases going towards G.
# - Once again, we can infer **grade** as an important parameter.

# In[84]:


plt.figure(figsize=(18,11));
segment_uni('sub_grade');


# #### Inference :
# 
# - There is a high difference between the default and non-default rate for better grades like A,B and steeply decreases going towards G.
# - Once again, we can infer **sub_grade** as an important parameter.

# In[85]:


# plt.figure(figsize=(12,7));
# segment_uni('purpose',90);


# In[86]:


# plt.figure(figsize=(12,7));
# segment_uni('verification_status');


# In[87]:


# plt.figure(figsize=(12,7));
# segment_uni('pub_rec_bankruptcies');


# # Univariate Analysis - Quantitative features

# In[88]:


# This user-defined function plots the distribution of target column, and its boxplot against loan_status column
def plot_distribution(var):
    plt.figure(figsize=(17,9))
    plt.subplot(1, 2, 1)
    ax = sns.histplot(data=loan_df, x=var, kde=True)
    plt.subplot(1, 2, 2)
    sns.boxplot(x=var, y= 'loan_status', data=loan_df)
    plt.show()


# In[89]:


plot_distribution('int_rate')


# In[90]:


loan_df.groupby('loan_status')['int_rate'].describe()


# In[91]:


cut_labels_4 = ['low','medium','high']
loan_df['Interest Rate'] = pd.cut(loan_df['int_rate'], bins=np.linspace(loan_df['int_rate'].min(), loan_df['int_rate'].max(), 4), labels=cut_labels_4)
loan_df['Interest Rate'].value_counts()


# In[92]:


plt.figure(figsize=(12,7));
default_rate_per_var('Interest Rate');


# #### Inference :
# 
# - For defaulters interest rate in Inter-Quartile range is high compared to non-defaulters.
# - So, **int_rate** is an **important** driving parameter.
# 
# - Also for low interest rates, the chance of defaulting is less and vice-versa.

# In[93]:


plot_distribution('installment')


# #### Inference :
# 
# - There is similar quartiles for both the deafult and non-defaulters.
# - We wont consider installment.

# In[94]:


plot_distribution('annual_inc')


# There are outliers in the income variable.
# This should be cleaned.

# #### Removing the outliers

# In[95]:


loan_df_inc = loan_df[np.abs(loan_df.annual_inc-loan_df.annual_inc.mean()) <= (3*loan_df.annual_inc.std())].copy()


# In[96]:


loan_df_inc.shape


# In[97]:


loan_df_inc['Annual Income'] = pd.cut(loan_df_inc['annual_inc'], bins=5)
loan_df_inc['Annual Income'].value_counts()


# In[98]:


plt.figure(figsize=(16,8));
default_rate_per_var('Annual Income', df=loan_df_inc);


# #### Inference :
# 
# - The % of defaulted loans is higher for lower to mid salary range.
# - So, **Annual Income** is an important factor.

# In[99]:


plot_distribution('dti')


# In[100]:


plot_distribution('open_acc')


# In[101]:


plot_distribution('revol_bal')


# In[102]:


plot_distribution('total_acc')


# In[103]:


plt.figure(figsize=(12,7));
default_rate_per_var('inq_last_6mths');


# In[104]:


plt.figure(figsize=(16,9));
default_rate_per_var('open_acc');


# In[105]:


loan_df['emp_length'].value_counts().sort_index()


# In[106]:


plt.figure(figsize=(16,8));
default_rate_per_var('emp_length');


# In[107]:


loan_df['revol_util'].describe()


# In[108]:


cut_labels_4 = ['low','medium','high','very high']
loan_df['Revolving Utilization'] = pd.cut(loan_df['revol_util'], bins=4, labels=cut_labels_4)
loan_df['Revolving Utilization'].value_counts().sort_index()


# In[109]:


plot_distribution('revol_util')


# In[110]:


plt.figure(figsize=(16,8));
default_rate_per_var('Revolving Utilization');


# #### Inference :
# 
# - The high and very high revolving utilization cases are more in case of defaulters.
# - Also the median is higher for defaulters.
# - So **Revolving utilization** is an important factor.

# In[111]:


# loan_df.addr_state.value_counts(normalize=True).head(10)*100
plt.figure(figsize=(20,10));
default_rate_per_var('addr_state', head=10);


# #### Inference :
# 
# - The distribution is for defaulter does not vary much with the entire portfolio.
# - So we will not consider **addr_state**

# In[112]:


loan_df['earliest_cr_line_year'] = loan_df['earliest_cr_line'].dt.year
loan_df.loc[loan_df['earliest_cr_line_year'] > 2011, 'earliest_cr_line_year' ].unique()
loan_df.loc[loan_df['earliest_cr_line_year'] > 2011 , 'earliest_cr_line_year'] = loan_df['earliest_cr_line_year'] - 100
sns.histplot(loan_df['earliest_cr_line_year'])


# In[113]:


plt.figure(figsize=(20,10));
default_rate_per_var('earliest_cr_line_year');


# #### Inference :
# 
# - The distribution is for defaulter does not vary much with the entire portfolio.
# - So we will not consider **credit line year**

# # Bivariate Analysis

# In[114]:


def bivariate(x_axis, y_axis, xlabel_rotation=0, legend_flag=False, est='mean'):
    if est == 'count_pct':
        ax = sns.barplot(x = x_axis,  y=y_axis, hue = 'loan_status', order= sorted(list(set(loan_df[x_axis]))), data = loan_df, estimator=lambda x: len(x) / len(loan_df) * 100 )
    else:
        ax = sns.barplot(x = x_axis,  y=y_axis, hue = 'loan_status', order= sorted(list(set(loan_df[x_axis]))),data = loan_df, estimator=np.mean )
    
    #plt.title(figure_title)
    plt.xticks(rotation = xlabel_rotation)
    #plt.xlabel(xlabel, labelpad = 15)
    #plt.ylabel(ylabel, labelpad = 10)
    
    if legend_flag:
        plt.legend(loc = legend)


# In[115]:


plt.figure(figsize=(20,9));
bivariate(x_axis='addr_state', y_axis='loan_amnt' );


# - State WY: Wyoming has the the highest average loan amount that was charged off. Can be investigated.

# In[116]:


plt.figure(figsize=(12,5));
bivariate(x_axis='grade', y_axis='revol_util' );


# #### Inference :
# 
# - The revolving utilization increases with the loans with lower grades.
# - So **Revolving utilization** is an important factor.

# In[117]:


plt.figure(figsize=(20,9))
sns.boxplot(x="grade", y="revol_util", hue="loan_status", data=loan_df, order= sorted(list(set(loan_df['grade']))),palette="Paired");


# In[118]:


plt.figure(figsize=(12,5));
bivariate(x_axis='term', y_axis='loan_amnt' );


# - The assumption made during univariate analysis is more evident with this plot. Higher loan amount are associated with longer terms and see higher Charge Offs.

# In[119]:


# 5.14 Define Correlation

# Correlation of loan_status = Charged Off
defaulted_df = loan_df[loan_df.loan_status == 'Charged Off'].corr().drop(labels = {'in_default_flg'})

defaulted_df.dropna(axis = 1, how = 'all', inplace = True)

defaulted_df


# In[120]:


ax = sns.clustermap(defaulted_df, annot=True, center=0, linewidths=.75, figsize=(22,21), cbar_pos=(.12, .32, .03, .2),);
ax.ax_row_dendrogram.remove()


# ## Conclusion:
# 
# #### Major variables to consider for loan prediction:
# 
#         1. Grade & Sub-grade
#         2. Interest Rate 
#         3. Term
#         4. Revolving Utilisation
#         5. Purpose
#         
# #### Purpose - Debt Consolidation
# #### State - WY- Wyoming
# #### Term - 60 months
# #### Grade - D,E,F,G are riskier along with its sub grades.
# #### Interest rate - On higher side

# In[ ]:




