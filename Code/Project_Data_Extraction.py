#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import os
import time


# In[13]:


def data_reader(excelPaths):
    dataset = []
    for key, value in excelPaths.items():
        for path in value[0]:
            print(path)
            
            x = pd.read_excel(path, sheet_name=['Mieterliste', 'Internationales Verfahren DCF'], header=None)
            
            #Time of the last modification in seconds
            ti_m = os.path.getmtime(path)
            #Time of the last modification standart
            m_ti = time.ctime(ti_m)
            
            features = collect_data(x)
            features.insert(0, key)
            features.append(ti_m)
            features.append(m_ti)
            dataset.append(features)
            
        #AMP - average market rent
        #Wohnfläche - LivingArea
        #Gewerbefläche - CommercialArea
        #JNKM  - Annual net cold rent - ANCA
        #LM_Timecode - Last Modified timecode
        #LM_date - Last Modified standart
        #Instandhaltungsrücklage - Maintenance reserve
        #Capex - Capital expenditure
        df = pd.DataFrame(dataset, columns =['City', 'Address', 'LivingArea', 'CommercialArea', 
                                             'ANCA', 'DCF_Price', 'AMP', 'MaintenanceReserve', 'Capex',
                                             'LM_Timecode', 'LM_date']) 
        writeFiles(df, key)
    return dataset
    
#Function to extract data from the Excel file based on the Excel sheet name and cell number        
def collect_data(excelFile):
    infos = []
    
    infos.append(str(excelFile['Internationales Verfahren DCF'].loc[1, 3]))          #Adresse
    infos.append(excelFile['Mieterliste'].loc[26, 8])                                #Wohnfläche
    infos.append(excelFile['Mieterliste'].loc[27, 8])                                #Gewerbefläche
    infos.append(excelFile['Internationales Verfahren DCF'].loc[82, 3])              #JNKM
    infos.append(excelFile['Internationales Verfahren DCF'].loc[12, 8])              #DCF Preis
    infos.append(excelFile['Internationales Verfahren DCF'].loc[26, 8])              #m²-Miete Marktüblich
    infos.append(excelFile['Internationales Verfahren DCF'].loc[48, 8])              #Instandhaltungsrücklage
    infos.append(excelFile['Internationales Verfahren DCF'].loc[50, 8])              #Capex
    
    return infos

#Objects with wrong valuations
exceptions = ['Commercial_Valuation_Schönebecker Straße 38, 28759 Bremen.xlsm', 
              'Commercial_Valuation_2019_v5.5 - Bornaische Str. 130, 04279 Leipzig - neu.xlsm',
              'Commercial_Valuation_2019_v5_Otto-Hahn-Straße 39, 47167 Duisburg.xlsm',
              'Commercial_Valuation_Template Osterfeldstraße 26 45886 Gelsenkirchen.xlsm',
              'Commercial_Valuation_2019_Karlsruher Straße 82, 69126 Heidelberg.xlsm',
              'Commercial Valuation Heinrich-Zille-Straße 6, 74078 Heilbronn.xlsm',
              'Commercial_Valuation_2019_Johanniterstraße 17-19, 67547 Worms.xlsm',
              'Commercial Valuation Melbeckstraße 18, 42655 Solingen.xlsm',
              'Commercial Valuation Melbeckstraße 18, 42655 Solingen (1).xlsm',
              'Commercial Valuation Katternberger Str. 148, 42655 Solingen.xlsm',
              'Commercial Valuation  Katternberger Straße 150, 42655 Solingen.xlsm',
              'Commercial_Valuation_Template Schützenstraße 102, 42655 Solingen.xlsm',
              'Commercial Valuation Adlerstraße 13, 42655 Solingen.xlsm',
              'Commercial Valuation Neuenweiherstraße, 91056 Erlangen.xlsm']

def roots(city, exceptions):
    counter = 0
    files   = []
    rootDir = 'G:\\Meine Ablage\\Valuation - Commercial\\' + city + '\\'
    print(city)
    for dirName, subdirList, fileList in os.walk(rootDir):
        if (('Datenraum' not in dirName) and ('Briefing' not in dirName) and 
            ('Projektierung' not in dirName) and ('2Pager' not in dirName) and 
            ('Residualwertverfahren' not in dirName)):
            for fname in fileList:
                if ((exceptions[0] or exceptions[1] or exceptions[2] or exceptions[3] or
                     exceptions[4] or exceptions[5] or exceptions[6]  or exceptions[7] or
                     exceptions[8] or exceptions[9] or exceptions[10] or exceptions[11] or
                     exceptions[12] or exceptions[13]) in fname):
                    continue
                if (('Commercial' in fname) and ('Valuation' in fname) and ('.xlsm' in fname) and 
                    (' ' in fname) and ('Projektierung' not in fname) and ('Briefing' not in fname) and 
                    ('.gsheet' not in fname) and ('~' not in fname)):
                    files.append(dirName + '\\' + fname)
                    counter += 1
                    if counter % 50 == 0:
                        print(counter)
    return files 

#List of all cities, where we did valuations (not only in top-100)
def listOfcities(rootDir_):
    subdirList = os.listdir(rootDir_)
    copyDirs   = subdirList.copy()
    newList = [x for x in copyDirs if (("." not in x) and ("New Folder" not in x)
                                      and ("0" not in x) and ("_" not in x))]
    return newList

#Preprocess the dataset with the biggest german cities
def germanCities(data):
    newGerCities = data.copy()
    
    #Some cities with missing information about the number of inhabitants
    specialCities1 = ['Offenbach', 'Weiden', 'Nienburg', 'Verden', 'Köthen', 'Rotenburg', 
                      'Karlstadt', 'Neustadt', 'Homberg', 'Garmisch-Partenkirchen']
    specCitPop1 = [131295, 43052, 31570, 27782, 24876, 22199, 14995, 10177, 13970, 27482]
    
    #Some cities where english city names need to be replaced by german names
    #or cities with incomplete names 
    specialCities2ENG = ['Frankfurt', 'Cologne', 'Munich', 'Nuremberg', 'Ludwigshafen', 'Mülheim', 'Offenbach', 'Esslingen']
    specialCities2GER = ['Frankfurt am Main', 'Köln', 'München', 'Nürnberg', 'Ludwigshafen am Rhein', 
                         'Mülheim an der Ruhr', 'Offenbach am Main', 'Esslingen am Neckar']
    
    #Some provinces where english names need to be replaced by german names
    specialAdmiName1 = ['Bavaria', 'North Rhine-Westphalia', 'Saxony', 'Lower Saxony', 'Saxony-Anhalt', 
                      'Mecklenburg-Western Pomerania', 'Thuringia', 'Rhineland-Palatinate', 'Hesse']
    
    specialAdmiName2 = ['Bayern', 'NRW', 'Sachsen', 'Niedersachsen', 'Sachsen_Anhalt', 'Mecklenburg_Vorpommern',
                        'Thüringen', 'Rheinland_Pfalz', 'Hessen']
    
    #Use these loops to make the corrections listed above
    for i in range(len(specialCities1)):
        newGerCities.loc[newGerCities.city == specialCities1[i], ['population', 'population_proper']] = specCitPop1[i]
        
    for i in range(len(specialCities2ENG)):
        newGerCities.loc[newGerCities.city == specialCities2ENG[i], 'city'] = specialCities2GER[i]    
    
    for i in range(len(specialAdmiName1)):
        newGerCities.loc[newGerCities.admin_name == specialAdmiName1[i], 'admin_name'] = specialAdmiName2[i]
    
    #Sort the cleaned dataframe by population 
    sortedGerCit = newGerCities.sort_values(by = 'population_proper', ascending=False, inplace=False)
    
    return sortedGerCit

#Use this function to add new object in the dictionary 
def add_element(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)  

#Create a dictionary with rooth to the objects
def dictOfroots(gerCitiesSorted, subdirList):
    d = {}
    for city in gerCitiesSorted.city:
        for cities in subdirList:
            if (city in cities):
                val  = roots(cities, exceptions) 
                add_element(d, cities, val)
    return d

#Write dataframe in the csv file
def writeFiles(data, city):
    path = 'C:\\Users\\User\\Desktop\\Projekt ML\\Interdisziplinäre Projektarbeit\\' + 'data' + city + '.csv'
    data.to_csv(path, index = False)   


# In[14]:


#all subdirectories in Valuation Commercial
rootDir    = 'G:\\Meine Ablage\\Valuation - Commercial\\'
subdirList = listOfcities(rootDir)

#dataset of german cities
gerCities       = pd.read_csv('Deutschland_Cities.csv', sep=',')
gerCitiesSorted = germanCities(gerCities)


# In[16]:


top1_10      = gerCitiesSorted[2:3]
dictionary1  = dictOfroots(top1_10, subdirList)


# In[ ]:


dataset1     = data_reader(dictionary1)


# In[26]:


top11_20     = gerCitiesSorted[10:20]
dictionary2  = dictOfroots(top11_20, subdirList)


# In[27]:


dataset2     = data_reader(dictionary2)


# In[28]:


top21_30     = gerCitiesSorted[20:30]
dictionary3  = dictOfroots(top21_30, subdirList)


# In[29]:


dataset3     = data_reader(dictionary3)


# In[30]:


top31_40     = gerCitiesSorted[30:40]
dictionary4  = dictOfroots(top31_40, subdirList)


# In[31]:


dataset4     = data_reader(dictionary4)


# In[32]:


top41_50     = gerCitiesSorted[40:50]
dictionary5  = dictOfroots(top41_50, subdirList)


# In[33]:


dataset5     = data_reader(dictionary5)


# In[55]:


top51_60     = gerCitiesSorted[50:60]
dictionary6  = dictOfroots(top51_60, subdirList)


# In[56]:


dataset6     = data_reader(dictionary6)


# In[57]:


top61_70     = gerCitiesSorted[60:70]
dictionary7  = dictOfroots(top61_70, subdirList)


# In[58]:


dataset7     = data_reader(dictionary7)


# In[4]:


top71_80     = gerCitiesSorted[70:80]
dictionary8  = dictOfroots(top71_80, subdirList)


# In[7]:


dataset8     = data_reader(dictionary8)


# In[8]:


top81_90     = gerCitiesSorted[80:90]
dictionary9  = dictOfroots(top81_90, subdirList)


# In[9]:


dataset9     = data_reader(dictionary9)


# In[10]:


top91_100    = gerCitiesSorted[90:100]
dictionary10 = dictOfroots(top91_100, subdirList)


# In[11]:


dataset10    = data_reader(dictionary10)

