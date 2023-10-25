#!/usr/bin/env python
# coding: utf-8

# # {Analyzing the Impact of COVID-19 Vaccination Campaigns on Public Health Outcomes}📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# How informed public health campaigns can affect the overall outcome of a pandemic or similar outbreak when it comes to receiving vaccination and proper care for said ailment. In my instances people can be ill informed about vaccinations or other forms of treatment for illnesses and have an adverse outlook of them when in reality they are often times just ill-equiped to make a proper judgement due to not having equal access to the information at hand to make their own informed decision.
# 
# Some questions that I will use to help fuel this inquiry are:
# 
# How have COVID-19 vaccination rates and the availability of vaccine doses evolved over time in various regions?
# 
# What is the relationship between vaccination rates and the incidence of COVID-19 cases in different demographic groups?
# 
# Are there any patterns or disparities in vaccination rates based on socioeconomic factors or geographic locations?
# 
# Though not every question above is not the "Project Question" itself, they are things to keep in mind that might add to the overall assessment of the issue at hand and provide insights as to why certain trends might be found.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# "How did various public health campaigns for the COVID-19 vaccine affect the outcome of public health in various socioeconomic areas?"

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# "I believe that there will be a reduction in spread and lower number of overall cases of COVID-19 in regions that had stronger public health campaigns that pushed for the vaccine as opposed to those who did not, notably this would be more likely to occur in areas that are of higher socioeconomic standing." 

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# The first is api from [covidactnow](https://covidactnow.org/?s=48083556). This has a lot of in depth and useful data available in both json & csv format.
# 
# The second source is from the [US Census Database](https://data.census.gov). This information is very useful in that it will provide the different demographics that will be used in the analysis.
# 
# The final source is from a csv file **covid_states_history.csv**. This file has information dating up until 2021 about various cases, deaths, etc. that will be further used to analyze the problem at hand.
# 
# The way I plan on relating the datasets is that I plan on merging the COVID-19 Vaccination data and Case data sets (API & CSV) in order to have common variables such as data, region, etc. to be used to analyze the relationship between COVID-19 cases and vaccination rates. Furthermore, I plan on relation the Socioeconomic Data (Database) to both the vaccine and case data using different regions as a common key to explore any patterns based on socioeconomic factors. 
# 
# 
# 
# 

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# 
# My approach to this project is that I will first get all of the data from the various sets imported, merged, and cleaned such that I can begin analyzing it. There are a few different ways that I plan to approach these datasets to analyze the problems at hand. I can potentially perform a time-series analysis using line charts that show vaccination rates & does availability by region over a set amount of time. I can use regression analysis to identify any statistically significant relationships between vaccination rates and COVID-19 case occurrence in terms of different demographic groups and visualize that with scatter plots. Finally, I can use heatmaps to visualize the different vaccine rates based on region and socioeconomic factors.  

# In[78]:


# Start your code here
from dotenv import load_dotenv

import os
import csv
import json

import pandas as pd
import numpy as np

from urllib.request import urlretrieve
from zipfile import ZipFile
from sqlalchemy import create_engine, text
from pymongo import MongoClient

import requests
from bs4 import BeautifulSoup

load_dotenv(override=True) 
pd.set_option('display.max_rows', 500)


# In[77]:


covid_history_df = pd.read_csv('covid_states_history.csv')

print(covid_history_df.head())



# In[76]:


census_education_df = pd.read_csv('census_education.csv')

print(census_education_df.head())


# In[75]:


census_health_df = pd.read_csv('census_health.csv')

print(census_health_df.head())


# In[80]:


census_income_df = pd.read_csv('census_income.csv')

print(census_income_df.head())


# In[79]:


census_population_df = pd.read_csv('census_population.csv')

print(census_population_df.head())


# In[81]:


api_key = os.getenv("COVID_ACTNOW_API_KEY")

url = f"https://api.covidactnow.org/v2/states.json?apiKey={api_key}"

response = requests.get(url)

#Request Check
if response.status_code == 200:
    data = response.json()

    #Pandas DataFrame
    covid_data = pd.DataFrame(data)
    
    print(covid_data.head())
else:
    print(f"Request failed with status code: {response.status_code}")


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# 
# [covidactnow.org](https://covidactnow.org/?s=48083556)
# 
# [US Census Database](https://data.census.gov)
# 
# [The Covid Tracking Project](https://covidtracking.com/data/download)
# 

# In[ ]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

