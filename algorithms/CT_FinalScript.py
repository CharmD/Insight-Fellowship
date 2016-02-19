"""
Created on Feb 5, 2016

@author: Charmaine Demanuele
charmaine.demanuele@gmail.com

Therapy Prospector
Predicting Rehabilitation Effects for Constant Therapy Users

"""

#%% Load data from Constant Therapy Database

import mysql.connector
from mysql.connector import errorcode

from scipy import stats
import numpy as np
import pylab as py 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.externals import joblib
import pickle

COMMA = ","

# the maximum length for the group concat string length (add zeros in case of Errors)
GROUP_CONCAT_MAX_LENGTH= '1000000'


"""
Python Class to store our TaskType information quiried from the database
"""
class TaskType:

    def __init__(self, id, maxLevel):
        self.id = id
        self.maxLevel = maxLevel  
        
    def __repr__(self):
        return "id: %d \t maxLevel: %d" % (self.id, self.maxLevel)
        
"""
Python Class to store our Response information quiried from the database
"""
class Response:

    def __init__(self, patientId, accuracies, latencies):
        self.patientId = patientId
        
        # in order to process them we need to first split those values by COMMA                 
        self.accuracies = [float(accuracy) for accuracy in accuracies.split(COMMA)]
        self.latencies = [float(latency) for latency in latencies.split(COMMA)]
        
    def __repr__(self):
        accuracies = [str(accuracy) for accuracy in self.accuracies]
        latencies = [str(latency) for latency in self.latencies]
        return "patientId: %d\naccuracies: %s\nlatency: %s" % (self.patientId, COMMA.join(accuracies), COMMA.join(latencies))
    
    def __cmp__(self, other):
        if hasattr(other, 'patientId'):
            return self.patientId.__cmp__(other.patientId) 
            
            
    def zero_slope(self, data, chunksize, max_slope = .001):
        midindex = chunksize / 2
        for index in xrange(len(data)/chunksize):
            start = (index * chunksize)
            chunk = data[start + 1 : start + chunksize, :]
            dx, dy = abs(chunk[0] - chunk[-1])
            #print dy, dx, dy / dx
            if 0 <= dy/dx < max_slope:
                return chunk[midindex]
        return []            
    
    # this function smooths the data with a window = 10% of the length of the
    # data OR a window of 10samples if the latter is <10
    # then an 8th order polynomial is fitted to the smoothed data (for latency)
    # and the minimum latency value and index are defined
    def computeLatencyAggregate(self, listVal):
        
        responseCount = len(listVal)
        winLen = int(.1 * responseCount)
        if winLen < 9:
            winLen = 9        
        
        mAvg = pd.rolling_mean(np.array(listVal[3:]), winLen + 1) # smoothing window at 10% of data
        ydata = mAvg[winLen:]
        xdata = range(0,len(ydata))
        z = np.polyfit(xdata, ydata, 8) #fit polynomial of order 8
        f2  = np.poly1d(z)
        x = np.linspace(np.min(xdata), np.max(xdata), len(xdata)*2)
        ynew = f2(x)
    
        LatencyValPlateau = np.min(ynew)  
        if LatencyValPlateau == None:
            LatencyValPlateau = np.NaN
            LatencyIndPlateau = np.NaN
        else:   
            location = np.where(ynew == LatencyValPlateau)
            LatencyIndPlateau = location[0][0]/2

        
        return ((mAvg[-1] - mAvg[winLen])/mAvg[winLen] * 100,           # compute % improvement
                (mAvg[-1] - mAvg[winLen]),                              # compute absolute improvement
                 mAvg[winLen],                                          # baseline level
                 mAvg[-1],                                              # end level
                 np.nanmean(pd.rolling_var(np.array(listVal[3:]), winLen + 1)), #mean variance
                 LatencyIndPlateau, #min index for latency
                 LatencyValPlateau);  #min latency value       
 
    # this function smooths the data with a window = 10% of the length of the
    # data OR a window of 10samples if the latter is <10
    # then an 4th order polynomial is fitted to the smoothed data (for accuracy)
    # and the accuracy plateau is defined
    def computeAccuracyAggregate(self, listVal):
        
        responseCount = len(listVal)
        winLen = int(.1 * responseCount)
        if winLen < 9:
            winLen = 9        

        mAvg = pd.rolling_mean(np.array(listVal[3:]), winLen + 1) # smoothing window at 10% of data
        ydata = mAvg[winLen:]
        xdata = range(0,len(ydata))
        z = np.polyfit(xdata, ydata, 4) #fit polynomial of order 4
        f2  = np.poly1d(z)
        x = np.linspace(np.min(xdata), np.max(xdata), len(xdata)*8)
        ynew = f2(x)
        arrayList = list()
        index = 0
        while index < len(x):
            arrayList.append((x[index], ynew[index]))
            index += 1
        data = np.array(arrayList)
        chunksize = int(len(data)/20)
        val = self.zero_slope(data, chunksize, max_slope = .001)
        if len(val) == 0:
            val = [np.NaN, np.NaN]
        
        accuracyIndPlateau = val[0] 
        accuracyValPlateau = val[1] 

        return ((mAvg[-1] - mAvg[winLen])/mAvg[winLen] * 100,           # compute % improvement
                (mAvg[-1] - mAvg[winLen]),                              # compute absolute improvement
                 mAvg[winLen],                                          # baseline level
                 mAvg[-1],                                              # end level
                 np.nanmean(pd.rolling_var(np.array(listVal[3:]), winLen + 1)), #mean variance  
                 accuracyIndPlateau,   #plateau accuracy index
                 accuracyValPlateau);  #plateau accuracy value    
        
    def toMatrixRow(self):
        
        responseCount = len(self.accuracies)
        if responseCount < 33:
            return [0, 0, 0, 0, 0, 0, 0, 0] #we have to do this because there's
            #a problem in the database in column completed_task_count
        
        matrixRow = list()
        matrixRow.append(self.patientId)
        
        y = self.latencies        
        tup = self.computeLatencyAggregate(y)        
        matrixRow.append(tup[0]*-1) # improveLatenciesPercent
        matrixRow.append(tup[1]*-1) # improveLatenciesAbsolute
        matrixRow.append(tup[2]) # baselineLatencies
        matrixRow.append(tup[3]) # endLatencies
        matrixRow.append(tup[4]) # meanVarLatencies 
        matrixRow.append(tup[5]) # min latency index
        matrixRow.append(tup[6]) # min latency value

        y = self.accuracies
        tup = self.computeAccuracyAggregate(y)
        matrixRow.append(tup[0]) # improveAccuraciesPercent
        matrixRow.append(tup[1]) # improveAccuraciesAbsolute
        matrixRow.append(tup[2]) # baselineAccuracies
        matrixRow.append(tup[3]) # endAccuracies 
        matrixRow.append(tup[4]) # meanVarAccuracies         
        matrixRow.append(tup[5]) # plateau accuracy index
        matrixRow.append(tup[6]) # plateau accuracy value
        
        matrixRow.append(responseCount)        
        return matrixRow

"""
Python Class to store our Patient Information quiried from the database
and create categorical variables to feed in the machine learning model
"""        
class Patient:
        
    def __init__(self, patientId, conditionSince, city, state, country, gender, age, disorders, disorderCount, deficits, deficitCount):
        
        EMPTY_STRING = ""
        DEFAULT_AGE = 0        
        
        self.patientId = patientId
                
        self.conditionSince = conditionSince
        if(conditionSince == None):
            self.conditionSince = EMPTY_STRING       
        
        self.city = city
        if(city == None):
            self.city = EMPTY_STRING 
            
        self.state = state
        if(state == None):
            self.state = EMPTY_STRING 
            
        self.country = country
        if(country == None):
            self.country = EMPTY_STRING 
            
        self.gender = gender
        if(gender == None):
            self.gender = EMPTY_STRING
            
        self.age = age
        if(age == None):
            self.age = DEFAULT_AGE
            
        self.disorders = disorders
        self.disorderCount = disorderCount
        
        self.deficits = deficits
        self.deficitCount = deficitCount
        
        
    def __repr__(self):
        return ("patientId: %d \t conditionSince: %s \t city: %s \t state: %s \t country: %s \t gender: %s \t age: %d \t disorders: %s \t disorderCount: %d  \t deficits: %s \t deficitCount: %d" 
        % (self.patientId, self.conditionSince, self.city, self.state, self.country, self.gender, self.age, self.disorders, self.disorderCount, self.deficits, self.deficitCount))
    
    
    def __cmp__(self, other):
        if hasattr(other, 'patientId'):
            return self.patientId.__cmp__(other.patientId)        
        
        
    def convertConditionSince(self):
        
        conditionArray = [0, 0, 0, 0, 0, 0, 0] 
        if self.conditionSince == u'6m':
            conditionArray[0] = 1
        elif self.conditionSince == u'1y':
            conditionArray[1] = 1
        elif self.conditionSince == u'2y':
            conditionArray[2] = 1
        elif self.conditionSince == u'5y':
            conditionArray[3] = 1
        elif self.conditionSince == u'10y':
            conditionArray[4] = 1
        elif self.conditionSince ==  u'>10y':
            conditionArray[5] = 1
        else:
            conditionArray[6] = 1
        return conditionArray
            
            
    def convertGender(self):
        
        genderArray = [0, 0, 0]
        genderLower = self.gender.lower()
        if genderLower.startswith('f'):                                    # female
            genderArray[0] = 1
        elif genderLower.startswith('m') or genderLower.startswith('b'):   # male OR boy
            genderArray[1] = 1        
        else:
            genderArray[2] = 1  #other
        return genderArray
       
            
    def convertDeficits(self):
        
        deficitList = sorted([int(deficit) for deficit in self.deficits.split(COMMA)])
        
        deficitArray = [0, 0, 0, 0, 0, 0, 0, 0, 0]        
        for deficit in deficitList:
            
            index = deficit - 1
            if index == 9998:
                deficitArray[-1] = 1            
            else:
                deficitArray[index] = 1
                
        return deficitArray
        
        
    def convertDisorders(self):
        
        disorderList = sorted([int(disorder) for disorder in self.disorders.split(COMMA)])
        
        disorderArray = [0, 0, 0, 0, 0, 0, 0]        
        for disorder in disorderList:
            
            index = disorder - 1
            if index == 9998:
                disorderArray[-1] = 1 
            else:
                disorderArray[index] = 1
                
        return disorderArray
        
        
    def toMatrixRow(self):        
        
        matrixRow = list()
        matrixRow.append(self.patientId)
        matrixRow.extend(self.convertConditionSince())
        matrixRow.extend(self.convertGender())
        matrixRow.append(self.age)
        matrixRow.extend(self.convertDisorders())
        matrixRow.append(self.disorderCount)
        matrixRow.extend(self.convertDeficits())
        matrixRow.append(self.deficitCount)
        return matrixRow
       
    
"""
Set up the database
"""      

def ResultIter(cursor, arraysize = 1000):
    
    'An iterator that uses fetchmany to keep memory usage down'
    while True:
        results = cursor.fetchmany(arraysize)
        if not results:
            break
        for result in results:
            yield result  

# set up database configuration: THIS IS NOT THE PASSWORD! CHANGE ACCORDINGLY
config = {
    'user': 'aa',
    'password': 'aa',
    'host': 'aa',
    'port': 'aa',
    'database': 'aa'
}

# open database connection
cnx = None
try:
    cnx = mysql.connector.connect(**config)
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)
    raise

# first get the task types and their max levels using a SQL query
query = "SELECT id, max_level FROM constant_therapy.task_types where is_active = 1"
cur = cnx.cursor()
cur.execute(query)

taskTypes = []  # create a new empty list for taskTypes
rows = cur.fetchall()
for row in rows:
    taskTypes.append(TaskType(row[0], row[1]))


# impose these constraints when running the program for all task types and levels
# the minimum number of patients we need per task type and task level
MIN_PATIENT_COUNT= 500

# the maximum number of disorders a patient can have
MAX_DISORDER_COUNT = 7

# the maximum number of deficits a patient can have
MAX_DEFICIT_COUNT = 9

query = "SET SESSION group_concat_max_len = {0}".format(GROUP_CONCAT_MAX_LENGTH)
cur = cnx.cursor()                  
cur.execute(query)

     

#%%###################################################################

"""
The following sections can be executed by two loops (one over task type and 
another over difficulty level)
The two loops are currently commented out to investigate certain task types 
and task levels in detail """

# in a loop for a given task type id and task level
# for taskType in taskTypes:
taskType  = taskTypes[43] 

# loop over task level: start from 1 and go Max level Range(1: maxLevel + 1)
# for level in range(1, taskType.maxLevel + 1):

level = 2;
     
print("current level: %d" % (level)) 

 # first get the patient ids we care about
query = """SELECT patient_id, SUM(completed_task_count) ctc
              FROM sessions s
              JOIN users u ON u.id = s.patient_id
              JOIN ct_customer.customers c on u.id = c.user_id
              WHERE c.created_date >= '2014-07-01'          -- users created after July 2014
                  AND (u.is_demo = 0 OR u.is_demo IS NULL)  -- exclude demo users
                  AND s.type = 'SCHEDULED'                  -- only 'SCHEDULED' patients
                  AND task_type_id = {0}
                  AND task_level = {1}
              GROUP BY patient_id
              HAVING ctc >= 33                              -- patents who have done more than 33 tasks of a given type and level
              ORDER BY task_type_id, task_level""".format(taskType.id, level)
cur = cnx.cursor()                  
cur.execute(query)
 
patientIds = []    # create a new empty list for patients id(s)
for row in cur:
    patientIds.append(row[0])
 
#check the number of patients returned for the task type and task level
#if(len(patientIds) < MIN_PATIENT_COUNT):
    #break    
    #return
     
commaSeperatedPatientList = COMMA.join([str(patient) for patient in patientIds])
 
#######################################################################
     
# get the patient features for filtered patient_id(s)
query = """SELECT user_id, condition_since, city, state, country, gender, (2016 - birth_year) AS age,
                   GROUP_CONCAT(DISTINCT(disorder_id)) disorders, count(DISTINCT(disorder_id)) disorderCount,
                   GROUP_CONCAT(DISTINCT(deficit_id)) deficits, count(DISTINCT(deficit_id)) deficitCount
              FROM ct_customer.customers c
              INNER JOIN ct_customer.customers_to_disorders d ON d.customer_id = c.id
              INNER JOIN ct_customer.customers_to_deficits f ON f.customer_id = c.id
              WHERE c.user_id IN ({0})
              GROUP BY d.customer_id
              HAVING disorderCount <= {1}
                  AND deficitCount <= {2}""".format(commaSeperatedPatientList, MAX_DISORDER_COUNT, MAX_DEFICIT_COUNT)
cur = cnx.cursor()
     # print("done with patient info query")
     
cur.execute(query)
patients = []  # create a new empty list for patients
for row in ResultIter(cur):
    patients.append(Patient(row[0], row[1], row[2], row[3], row[4], 
                             row[5], row[6], row[7], row[8], row[9], row[10])) 
 
#######################################################################
         
# get responses based on the patients filtered above# 
query = """SELECT r.patient_id, GROUP_CONCAT(r.accuracy) accuracies, GROUP_CONCAT(r.latency) latencies
             FROM responses r
             JOIN sessions s on r.session_id = s.id
             WHERE r.task_type_id = {0}
               AND r.task_level = {1}
    		 AND s.type = 'SCHEDULED'
    		 AND r.skipped = 0
               AND r.patient_id IN ({2})
             GROUP BY r.patient_id
             ORDER BY r.timestamp""".format(taskType.id, level, commaSeperatedPatientList)
cur = cnx.cursor(buffered = True)
# print("done with response query")
 
cur.execute(query)
responses = []  # create a new empty list for responses
for row in ResultIter(cur):
    responses.append(Response(row[0], row[1], row[2]))
                                                       
#break  # stop at first level       
#break  # stop at first task type   

# done with cursor
cur.close()             

# done with database
cnx.close()


#######################################################################

#Convert patient information into a matrix of features to feed into machine 
#learning algorithm
patientDoubleList = list()
for patient in sorted(patients):
     patientDoubleList.append(patient.toMatrixRow()) 
patientMatrix = np.array(patientDoubleList)

responseDoubleList = list()
for response in sorted(responses):
     responseDoubleList.append(response.toMatrixRow()) 
responseMatrix = np.array(responseDoubleList)

patientCompleteList = list()
for patientMatrixRow in patientDoubleList:
     patientId = patientMatrixRow[0]
     
     currentPatientRow = list(patientMatrixRow)
     removeResponseRow = None     
     for responseMatrixRow in responseDoubleList:
         if responseMatrixRow[0] == patientId:
             currentPatientRow.extend(responseMatrixRow[1:])
             removeResponseRow = responseMatrixRow
             break
     
     if removeResponseRow:
         responseDoubleList.remove(removeResponseRow)
         patientCompleteList.append(currentPatientRow)
 

patientMatrixOrig = np.array(patientCompleteList)

#delete column related to Apraxia from the very beginning
#since the current database has no or close to no patients with this disorder
patientMatrix = np.delete(patientMatrixOrig, 17, 1)



#%% Understanding the data through visualization

#Plot age distributions and replace missing AGE values by the median age for 
#that disorder
fig, ax = plt.subplots(6,1,figsize=(4,12))
i=0
medianAge = []
disorderLabel = ['Age: TBI', 'Stroke', 'Aphasia', 'Dyslexia', 'Dementia', 'Other'];
AGE = patientMatrix[:,11]
for column in range(12,18): #disorder 1-6
    a = patientMatrix[:,column] #disorder column
    medianAge.append(np.median(patientMatrix[a>0,11]))
    age = patientMatrix[a>0,11]
    age[np.where(age==0)] = medianAge[i]
    AGE[a>0] = age
    ax[i].hist(age, 30)
    ax[i].set_title(disorderLabel[i])
    ax[i].set_ylabel('count')
    i+=1  
patientMatrix[:,11] = AGE;
plt.tight_layout() 

#plot latency improvement distribution for each group
disorderLabel = ['Latency: TBI', 'Stroke', 'Aphasia', 'Dyslexia', 'Dementia', 'Other'];
fig, ax = plt.subplots(6,1,figsize=(6,12))
i=0
medianLatencies = []
meanLatencies = []
modeLatencies = []
for column in range(12,18): #disorders 1-6
    a = patientMatrix[:,column] #disorder column
    groupLatencies = patientMatrix[a>0,30]  
    medianLatencies.append(np.median(groupLatencies))
    meanLatencies.append(np.mean(groupLatencies))
    modeLatencies.append(stats.mode(np.round(groupLatencies)))
    ax[i].hist(groupLatencies, 20)
    ax[i].set_title(disorderLabel[i])
    ax[i].set_ylabel('count')
    i+=1
plt.tight_layout()    
plt.show()

#plot accuracy improvement distribution for each group
disorderLabel = ['Accuracy: TBI', 'Stroke', 'Aphasia', 'Dyslexia', 'Dementia', 'Other'];
fig, ax = plt.subplots(6,1,figsize=(6,12))
i=0
medianAccuracies = []
meanAccuracies = []
modeAccuracies = []
for column in range(12,18): #disorder 1-6
    a = patientMatrix[:,column] #disorder column
    groupAccuracies = patientMatrix[a>0,37]  
    medianAccuracies.append(np.median(groupAccuracies))
    meanAccuracies.append(np.mean(groupAccuracies))
    modeAccuracies.append(stats.mode(np.round(groupAccuracies)))
    ax[i].hist(groupAccuracies, 20)
    ax[i].set_title(disorderLabel[i])
    ax[i].set_ylabel('count')
    i+=1
plt.tight_layout()    
plt.show()
    
#Observe which columns have many nonzero elements
nonzeros = []
for column in range(patientMatrix.shape[1]):
    a = patientMatrix[:,column]
    nonzeros.append(sum(a>0))
print(nonzeros)


#find group means for behavioral data
meanBaselineAccuracy = []
steBaselineAccuracy = []
for column in range(12,18): #disorder 1-6
    a = patientMatrix[:,column] #disorder column
    group = patientMatrix[a>0,38] #index within patient's matrix
    steBaselineAccuracy.append(stats.sem(group[group>0]))
    meanBaselineAccuracy.append(np.nanmean(group))

meanEndAccuracy = []
steEndAccuracy = []
for column in range(12,18): 
    a = patientMatrix[:,column] 
    group = patientMatrix[a>0,39]  
    steEndAccuracy.append(stats.sem(group[group>0]))
    meanEndAccuracy.append(np.nanmean(group))

meanResponseCount = []
steResponseCount = []
for column in range(12,18): 
    a = patientMatrix[:,column] 
    group = patientMatrix[a>0,43]  
    steResponseCount.append(stats.sem(group[group>0]))
    meanResponseCount.append(np.nanmean(group))
    

meanPlateauAccuracy = []
stePlateauAccuracy = []
for column in range(12,18): 
    a = patientMatrix[:,column] 
    group = patientMatrix[a>0,42]  
    stePlateauAccuracy.append(stats.sem(group[group>0]))
    meanPlateauAccuracy.append(np.nanmean(group))

meanPlateauAccuracyInd = []
stePlateauAccuracyInd = []
for column in range(12,18): 
    a = patientMatrix[:,column] 
    group = patientMatrix[a>0,41]  
    stePlateauAccuracyInd.append(stats.sem(group[group>0]))
    meanPlateauAccuracyInd.append(np.nanmean(group))

meanBaselineLatency = []
steBaselineLatency= []
for column in range(12,18): #disorder 1-6
    a = patientMatrix[:,column] #disorder column
    group = patientMatrix[a>0,31]  
    steBaselineLatency.append(stats.sem(group[group>0]))
    meanBaselineLatency.append(np.nanmean(group))

meanEndLatency = []
steEndLatency= []
for column in range(12,18): #disorder 1-6
    a = patientMatrix[:,column] #disorder column
    group = patientMatrix[a>0,32]  
    steEndLatency.append(stats.sem(group[group>0]))
    meanEndLatency.append(np.nanmean(group))
    
meanMinLatency = []
steMinLatency= []
for column in range(12,18): #disorder 1-6
    a = patientMatrix[:,column] #disorder column
    group = patientMatrix[a>0,35]  
    steMinLatency.append(stats.sem(group[group>0]))
    meanMinLatency.append(np.nanmean(group))
 
meanMinLatencyInd = []
steMinLatencyInd= []
for column in range(12,18): #disorder 1-6
    a = patientMatrix[:,column] #disorder column
    group = patientMatrix[a>0,34]  
    steMinLatencyInd.append(stats.sem(group[group>0]))
    meanMinLatencyInd.append(np.nanmean(group))   


# Plot bar plots for accuracy, latency and response counts
disorderLabel = ['TBI', 'Stroke', 'Aphasia', 'Dyslexia', 'Dementia', 'Other'];
ind = np.arange(6) 
width = 0.35       # the width of the bars

fig, ax = plt.subplots(figsize=(10,3.7))
rects1 = ax.bar(ind, meanBaselineAccuracy, width, color = sns.xkcd_rgb["denim blue"], 
                yerr=steBaselineAccuracy, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
rects2 = ax.bar(ind + width, meanEndAccuracy, width, color = sns.xkcd_rgb["pale red"], 
                yerr=steEndAccuracy, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
ax.set_ylim([0.5, 1])
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('Accuracy at Start and End of Therapy', fontsize=16)
ax.set_xticks(ind + width)
ax.set_xticklabels(disorderLabel, fontsize=14)
ax.legend((rects1[0], rects2[0]), ('Start', 'End'), fontsize=14)
ax.set_xlim([-0.5, 6.5])

fig, ax = plt.subplots(figsize=(10,3.7))
rects1 = ax.bar(ind, meanBaselineLatency, width, color = sns.xkcd_rgb["medium green"], 
                yerr=steBaselineLatency, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
rects2 = ax.bar(ind + width, meanEndLatency, width, color = sns.xkcd_rgb["pale red"], 
                yerr=steEndLatency, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
ax.set_ylabel('Latency (s)', fontsize=14)
ax.set_title('Latency at Start and End of Therapy', fontsize=16)
ax.set_xticks(ind + width)
ax.set_xticklabels(disorderLabel, fontsize=14)
ax.legend((rects1[0], rects2[0]), ('Start', 'End'), fontsize=14)
ax.set_xlim([-0.5, 6.5])

fig, ax = plt.subplots(figsize=(10,3.5))
rects1 = ax.bar(ind, meanResponseCount, width, color='black', yerr=steResponseCount)
ax.set_ylabel('Response Count', fontsize=14)
ax.set_title('Response Count', fontsize=16)
ax.set_xticks(ind + width/2)
ax.set_xticklabels(disorderLabel, fontsize=14)
ax.set_xlim([-.5, 6])

width = 0.2       # reduce bars width
fig, ax = plt.subplots(figsize=(10,4))
rects1 = ax.bar(ind, meanResponseCount, width, color = sns.xkcd_rgb["black"], 
                yerr=steResponseCount, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
rects2 = ax.bar(ind + width, meanMinLatencyInd, width, color = sns.xkcd_rgb["medium green"], 
                yerr=steMinLatencyInd, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
rects3 = ax.bar(ind + width*2, meanPlateauAccuracyInd, width, color = sns.xkcd_rgb["red"], 
                yerr=stePlateauAccuracyInd, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
ax.set_ylabel('Response Count', fontsize=14)
ax.set_title('Response Count with 3 metrics', fontsize=16)
ax.set_xticks(ind + width*1.5)
ax.set_xticklabels(disorderLabel, fontsize=14)
ax.legend((rects1[0], rects2[0], rects3[0]), ('At End', 'Min Latency', 'Plateau Accuracy'), fontsize=14, loc='best')

fig, ax = plt.subplots(figsize=(10,4))
rects1 = ax.bar(ind, meanBaselineAccuracy, width, color = sns.xkcd_rgb["black"], 
                yerr=steBaselineAccuracy, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
rects2 = ax.bar(ind + width, meanPlateauAccuracy, width, color = sns.xkcd_rgb["medium green"], 
                yerr=stePlateauAccuracy, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
rects3 = ax.bar(ind + width*2, meanEndAccuracy, width, color = sns.xkcd_rgb["red"], 
                yerr=steEndAccuracy, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('Accuracy at start, plateau and end', fontsize=16)
ax.set_xticks(ind + width*1.5)
ax.set_xticklabels(disorderLabel, fontsize=14)
ax.legend((rects1[0], rects2[0], rects3[0]), ('Start', 'Plateau', 'End'), fontsize=14, loc='best')
ax.set_ylim([0.5, 1])
    
fig, ax = plt.subplots(figsize=(10,4))
rects1 = ax.bar(ind, meanBaselineLatency, width, color = sns.xkcd_rgb["black"], 
                yerr=steBaselineLatency, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
rects2 = ax.bar(ind + width, meanMinLatency, width, color = sns.xkcd_rgb["medium green"], 
                yerr=steMinLatency, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
rects3 = ax.bar(ind + width*2, meanEndLatency, width, color = sns.xkcd_rgb["red"], 
                yerr=steEndLatency, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
ax.set_ylabel('Latency (ms)', fontsize=14)
ax.set_title('Latency at start, minimum and end', fontsize=16)
ax.set_xticks(ind + width*1.5)
ax.set_xticklabels(disorderLabel, fontsize=14)
ax.legend((rects1[0], rects2[0], rects3[0]), ('Start', 'Min', 'End'), fontsize=14, loc='best')


#%% Data Visualization with Seaborn 

#Convert patientMatrix into a data frame 
df = pd.DataFrame({'CS6m':patientMatrix[:,1], 'CS1y':patientMatrix[:,2], 'CS2y':patientMatrix[:,3], 
'CS5y':patientMatrix[:,4], 'CS10y':patientMatrix[:,5], 'CS>10y':patientMatrix[:,6], 'CSOther':patientMatrix[:,7], 
'Male':patientMatrix[:,8], 'Female':patientMatrix[:,9], 'GenderOther':patientMatrix[:,10], 'Age':patientMatrix[:,11],
'TBI':patientMatrix[:,12], 'Stroke':patientMatrix[:,13], 'Aphasia':patientMatrix[:,14], 'Dyslexia':patientMatrix[:,15],
'Dementia':patientMatrix[:,16], 'Other Disorder':patientMatrix[:,17], 'Disorder Count':patientMatrix[:,18], 
'Reading':patientMatrix[:,19], 'Writing':patientMatrix[:,20], 'Speaking':patientMatrix[:,21], 'Comprehension':patientMatrix[:,22],
'Attention':patientMatrix[:,23], 'Memory':patientMatrix[:,24], 'Visual':patientMatrix[:,25], 'Problem Solving':patientMatrix[:,26],
'Other Deficits':patientMatrix[:,27], 'Deficit Count':patientMatrix[:,28], 
'Improve Latency Percent':patientMatrix[:,29], 'Improve Latency Absolute':patientMatrix[:,30], 'Baseline Latency':patientMatrix[:,31], 
'End Latency':patientMatrix[:,32], 'Var Latency':patientMatrix[:,33], 'Min Latency Index':patientMatrix[:,34], 'Latency Min':patientMatrix[:,35], 
'Improve Accuracy Percent':patientMatrix[:,36], 'Improve Accuracy Absolute':patientMatrix[:,37], 'Baseline Accuracy':patientMatrix[:,38], 
'End Accuracy':patientMatrix[:,39], 'Var Accuracy':patientMatrix[:,40], 'Plateau Accuracy Index':patientMatrix[:,41], 'Plateau Accuracy':patientMatrix[:,42],
'Response Count':patientMatrix[:,43]})

#observe what are the variables that correlate with Baseline Accuracy
df.corr()["Baseline Accuracy"]
  
#compute distribution and correlation plots 
with sns.plotting_context("notebook",font_scale=1.4):
    sns.jointplot(x="Baseline Accuracy", y="End Accuracy", data=df, kind="reg",  size=6);
    
with sns.plotting_context("notebook",font_scale=1.4):
    sns.jointplot(x="Response Count", y="End Accuracy", data=df,  kind="reg", size=6, color='black');

with sns.plotting_context("notebook",font_scale=1.4):
    sns.jointplot(x="Response Count", y="Improve Accuracy Absolute", data=df,  kind="reg", size=6, color='black');
    
with sns.plotting_context("notebook",font_scale=1.4):
    sns.jointplot(x="Baseline Accuracy", y="Baseline Latency", data=df, kind="reg",  size=6, color='green');    

with sns.plotting_context("notebook",font_scale=1.4):
    sns.jointplot(x="Baseline Accuracy", y="Improve Accuracy Absolute", data=df, 
                  kind="reg", color='red', size=6);
                                  
s = df[['End Accuracy', 'Baseline Accuracy', 'Improve Accuracy Absolute']];
with sns.plotting_context("notebook",font_scale=1.2):
    sns.pairplot(s, size=3)

s = df[['Male', 'Age', 'CS6m']];
with sns.plotting_context("notebook",font_scale=1.2):
    sns.pairplot(s, size=3)

s = df[['End Latency', 'Baseline Latency', 'Improve Latency Absolute', 'Response Count']];
sns.pairplot(s, size=2.5)

with sns.axes_style('white'), sns.plotting_context("notebook",font_scale=1.2):
    sns.jointplot("Baseline Accuracy", "Baseline Latency", data=df, kind='hex', color='green')



#%% Build linear regression models to predict End Accuracy

#NOTE: set random_state = None if you want the random samples to change each time.

from sklearn import linear_model
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score, train_test_split

lenPatientMatrix = len(patientMatrix)

temp = range(1,29)
selectFeatures = np.append(temp, [38, 43])
     
labels = ['CS6m', 'CS1y', 'CS2y', 'CS5y', 'CS10y', 'CS>10y', 'CSOther','Gender Male', 
  'Gender Female', 'Gender Other', 'Age', 'TBI', 'Stroke', 'Aphasia',
  'Dsylexia', 'Dementia', 'Other Disorder', 'Disorder Count', 'Reading', 'Writing',
  'Speaking', 'Comprehension', 'Attention', 'Memory', 'Visual',
  'Problem Solving', 'Other Deficits', 'Deficit Count', 
  'Baseline Accuracy', 'Response Count']
labels2 = np.array(labels)

X = patientMatrix[:,selectFeatures]
y = patientMatrix[:,39] #predicting end accuracy

#use 5-fold cross-validation 
kfold = cross_validation.KFold(lenPatientMatrix, n_folds = 3, shuffle = True, random_state = 42)
#for train_indices, test_indices in kfold:
#    print('Train: %s | test: %s' % (train_indices, test_indices))

#Loop to reduce one feature at a time
#Consider the average accuracy for a specific set of features
def score(features):
    allscores = []
    features = np.array(features)
    cX = X[:, features]
    print cX.shape
    for k, (train, test) in enumerate(kfold):
        regr = linear_model.LinearRegression(normalize = True)
        regr.fit(cX[train, :], y[train])
        allscores.append(regr.score(cX[test, :], y[test]))
        #print("[fold {0}] score: {1:.5f}".
        #      format(k, regr.score(cX[test, :], y[test])))
    return np.mean(allscores)
print()

#choose the top 15 features which are most predictive
initial_features = np.arange(X.shape[1])
while len(initial_features) > 15:
    feature_scores = []
    for i in range(len(initial_features)):
        excludedfeatures = np.copy(initial_features)
        excludedfeatures = np.delete(excludedfeatures, i)
        feature_scores.append(score(excludedfeatures))
    worst_feature = np.argmax(feature_scores)
    print "worst:", worst_feature, "rem.score:", np.max(feature_scores)
    excludedfeatures = np.copy(initial_features)
    initial_features = np.delete(excludedfeatures, worst_feature)
print initial_features

cX = X[:, initial_features]
allscores = []
for k, (train, test) in enumerate(kfold):
    regr = linear_model.LinearRegression(normalize = True)
    regr.fit(cX[train, :], y[train])
    allscores.append(regr.score(cX[test, :], y[test]))
print(allscores)

#Regression score on all the data
regr.fit(cX, y)
print(regr.coef_/np.max(regr.coef_))
regr.score(cX, y)

#plot of feature importance derived by cross validation
labels2[initial_features]
plt.figure()
x = range(0,len(initial_features)+1)
plt.plot(regr.coef_/np.max(regr.coef_), 'ro')
plt.title("Linear Regression to Predict End Accuracy", fontsize=16)
plt.ylabel("Normalized coefficients", fontsize=16); plt.xlabel("Predictors", fontsize=16)
plt.xticks(x, labels2[initial_features], rotation='vertical')
plt.show()
plt.xlim([-0.5, len(initial_features)])

#Consider these predictive features and split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X[:,initial_features], y, test_size=0.3, random_state=42)

# Run linear regression with kfold cross validation on the train set
kfold = cross_validation.KFold(len(X_train), n_folds=3, shuffle = True, random_state=42)
estimator = linear_model.LinearRegression(normalize = True)
estimator.fit(X_train, y_train) #fitting the linear regression 3 times 
score = cross_val_score(estimator, X_train, y_train, cv = kfold)
score.mean()

plt.figure()
x = range(0,X.shape[1]+1)
plt.plot(estimator.coef_/np.max(estimator.coef_), 'ro')
plt.title("Linear Regression to Predict End Accuracy", fontsize=16)
plt.ylabel("Normalized coefficients", fontsize=16); plt.xlabel("Predictors", fontsize=16)
plt.xticks(x, labels2[initial_features], rotation='vertical')
plt.show()
plt.xlim([-0.5, len(initial_features)])

#test the model on the test set
MSE = np.mean((estimator.predict(X_test)-y_test)**2)
print(estimator.score(X_test, y_test))

predY = estimator.predict(X_test)

#derive Pearson's Correlation between the actual and predicted values
modelDf = pd.DataFrame({'predY':predY, 'y':y_test}) 
with sns.plotting_context("notebook",font_scale=1.2):
    sns.jointplot(x="y", y="predY", data = modelDf, kind="reg")
    


#%% Using non-linear Random Forest Regression to Predict End Accuracy

from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV

X = patientMatrix[:,selectFeatures]
y = patientMatrix[:,39] #predicting end accuracy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Run random forest regression with kfold cross validation on the train set
kfold = cross_validation.KFold(len(X_train), n_folds=3, shuffle = True, random_state=None)
rfEstimator = RandomForestRegressor(random_state=0, n_estimators=50)
score = cross_val_score(rfEstimator, X_train, y_train, cv = kfold)
print("Random Forest Training Score = %f " %score.mean())

#using GridSearch not to optimize the parameters but to create an estimator object
rfEstimator = GridSearchCV(RandomForestRegressor(random_state=0, n_estimators=50), {},
                         cv = kfold)
rfEstimator.fit(X_train, y_train) #fitting the model n_folds times   

#grid score
rfEstimator.grid_scores_

#feature importance
featureImportance = rfEstimator.best_estimator_.feature_importances_

plt.figure()
x = range(0,len(selectFeatures)+1)
plt.plot(featureImportance, 'ro')
plt.title("Random Forest Regression Feature Importance", fontsize=16)
plt.ylabel("Feature Importance", fontsize=16); plt.xlabel("Predictors", fontsize=16)
plt.show()
plt.xlim([-0.5, len(selectFeatures)])
plt.xticks(x, labels2, rotation='vertical')

#the function immediately gives you the estimates for the best score
MSE = np.mean((rfEstimator.predict(X_test)-y_test)**2)
print("Random Forest Test Score = %f " %rfEstimator.score(X_test, y_test))

predY = rfEstimator.best_estimator_.predict(X_test)
#predY2 = estimator.predict(X_test) same as predY

modelDf = pd.DataFrame({'predY':predY, 'y':y_test}) 
with sns.plotting_context("notebook",font_scale=1.2):
    sns.jointplot(x="y", y="predY", data = modelDf, kind="reg")
    


#%% Exploratory Analysis

"""
Can we classify the 6 patient cohorts based on their behavioral data 
(accuracy and latency values, their improvement and variance) and patient 
characteristics (age, gender, deficits and time since diagnosis)? 
Gain insight into differential task effects on patient groups
"""

"""
Principal Component Analysis (PCA) applied to this data identifies the 
combination of attributes (principal components, or directions in the feature 
space) that account for the most variance in the data. 
Here we plot the different samples on the 2 first principal components.
"""

from sklearn.decomposition import PCA

#treat disorders as your labels -> extract a vector that assigns a class
#to each of the patients based on their disorder
disorderClass = np.zeros(len(patientMatrix))
count = 1
for column in range(12,18): #disorder 1-6
    a = patientMatrix[:,column] #disorder column
    disorderClass[a>0] = count
    count+=1
    
#Matrix of patient's characteristics (not including disorder) & behavioral data     
temp = np.append(range(1,12), range(18,33))
temp2 = np.append(temp, range(37,40))
selectFeatures = np.append(temp2, 43) 
X_full = patientMatrix[:,selectFeatures]

pca = PCA(n_components=10, whiten=True)
pca.fit(X_full)
print(pca.explained_variance_ratio_) 

#determine the percentage of variance explained by each of the 10 components
py.figure()
py.plot(range(1,11), pca.explained_variance_ratio_)
py.title("PCA Explained Variance", fontsize=16)
py.xlabel("principal components", fontsize=14)
py.ylabel("ratio of explained variance", fontsize=14)

#transform X onto the first 2 independent components explaining
#more than 95% of the variance
pca = PCA(n_components=2, whiten=True)
X_t = pca.fit_transform(X_full)

disorderLabel = ['TBI', 'Stroke', 'Aphasia', 'Dyslexia', 'Dementia', 'Other'];
disorderColors = ['b', 'r', 'g', 'm', 'y', 'c']

# Reorder the labels to have colors matching the cluster results
py.figure()
for (c, (i, label)) in zip(disorderColors, enumerate(disorderLabel)):
    py.scatter(X_t[disorderClass == i+1, 0], X_t[disorderClass == i+1, 1], c = c, label = label)
py.legend(fontsize=12, loc="upper left")
py.show()
py.title('Principal Component Analysis', fontsize=16)
py.xlabel('1st PCA component', fontsize=14)
py.ylabel('2nd PCA component', fontsize=14)


#PCA doesn't do a v good job in this case!
#Let's first reduce the dimension to 10 by PCA, then apply
#unsupervised t-distributed Stochastic Neighbor Embedding
from sklearn.manifold import TSNE

pca = PCA(n_components=10, whiten=True)
X_t = pca.fit_transform(X_full)

model = TSNE(n_components=2)
x2d_TSNE = model.fit_transform(X_t)

py.figure()
for (c, (i, label)) in zip(disorderColors, enumerate(disorderLabel)):
    py.scatter(x2d_TSNE[disorderClass == i+1, 0], x2d_TSNE[disorderClass == i+1, 1], c = c, label = label)
py.legend(fontsize=12)
py.show()
py.title('Distributed Stochastic Neighbor Embedding', fontsize=16)
py.xlabel('1st TSNE component', fontsize=14)
py.ylabel('2nd TSNE component', fontsize=14)
#also doesn't work v well


"""
Linear Discriminant Analysis (LDA) tries to find attributes that 
account for the most variance between classes. 
LDA, in contrast to PCA, is a supervised method, 
using known class labels.
"""

#projecting onto two components
#from sklearn.lda import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_validation import cross_val_score, train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix

model = LDA(n_components=2)
x2d_lda=model.fit_transform(X_full, disorderClass.astype(int))
  
py.figure()
for (c, (i, label)) in zip(disorderColors, enumerate(disorderLabel)):
    py.scatter(x2d_lda[disorderClass == i+1, 0], x2d_lda[disorderClass == i+1, 1], c = c, label = label)
py.legend(fontsize=12)
py.show()
py.title('Linear Discriminant Analysis', fontsize=16)
py.xlabel('1st component', fontsize=14)
py.ylabel('2nd component', fontsize=14)
#We can see class separation - GREAT!

#supervised LDA projected onto 3 components
#Let's define classification accuracy
#split the data into training and test
X_train, X_test, y_train, y_test = train_test_split(X_full, disorderClass.astype(int), test_size=0.3, random_state=42)

model3 = LDA(n_components=3)
x3d_lda=model3.fit_transform(X_train, y_train)

fig = plt.figure(1, figsize=(4, 3), facecolor='white')
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
for (c, (i, label)) in zip(disorderColors, enumerate(disorderLabel)):
    ax.scatter(x3d_lda[y_train == i+1, 0], x3d_lda[y_train == i+1, 1], x3d_lda[y_train== i+1, 2], c = c, label = label)
ax.legend(fontsize=12, loc='upper left')
ax.set_xlabel('1st LD', fontsize=14)
ax.set_ylabel('2nd LD', fontsize=14)
ax.set_zlabel('3rd LD', fontsize=14)

classificationScoreTrain = model3.score(X_train, y_train)
print("Classification score on the training set is %f" %classificationScoreTrain)

y_pred = model3.predict(X_test) #gives you the predicted label for each sample
predictedProba = model3.predict_proba(X_test) #the probability of each sample to belong to each class
classificationScore = model3.score(X_test, y_test)
print("Classification score on the test data is %f" %classificationScore)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(disorderLabel))
    plt.xticks(tick_marks, disorderLabel, rotation=45, fontsize=12)
    plt.yticks(tick_marks, disorderLabel, fontsize=12)
    plt.tight_layout()
    plt.ylabel('True label',fontsize=14)
    plt.xlabel('Predicted label',fontsize=14)
    plt.grid(False)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()






#%% ########################################################################
"""

The following sections are not really necessary for further development
and may be deleted. I'm only including them for completeness.
 
"""

#%% THIS IS THE SIMPLIFIED LINEAR REGRESSION MODEL I USED FOR MY APP 
# PROTOTYPE for ACT Level 2: You may want to comment this out entirely

"""
Given the info above from linear regression and random forest regression (ran in R)
I can choose the following features:
Gender (male, female)
Deficit (reading, writing, problem solving and attention)
Condition since 6m, 1y, 2y, 5y, 10y, >10y
Baseline Accuracy

But to make it more general, I also added
Age and
Disorder type TBI, Stroke, Aphasia, Dyslexia, Dementia, Other Disorder

"""
from sklearn import linear_model
from sklearn import cross_validation

lenPatientMatrix = len(patientMatrix)

X = patientMatrix
y = patientMatrix[:,39] #predicting end accuracy

kfold = cross_validation.KFold(lenPatientMatrix, n_folds=3, shuffle = True, random_state=None)

#include important features condition since, age, deficit reading, writing,
#attention and problem solving
Rfeatures = np.append(range(1,7), [8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 23, 26, 38]) 

cX = X[:, Rfeatures]
allscores = []
allMSE = []
for k, (train, test) in enumerate(kfold):
    regr = linear_model.LinearRegression(normalize = True)
    regr.fit(cX[train, :], y[train])
    allscores.append(regr.score(cX[test, :], y[test]))
    allMSE.append(np.mean((regr.predict(cX[test, :])-y[test])**2))
print(allscores)
print(allMSE)
   
#plt.figure()
#x = range(0,len(Rfeatures)+1)
#plt.plot(regr.coef_/np.max(regr.coef_), 'ro')
#plt.title("Linear Regression to Predict End Accuracy", fontsize=16)
#plt.ylabel("Normalized coefficients", fontsize=16); plt.xlabel("Predictors", fontsize=16)
#plt.show()
#plt.xlim([-0.5, len(Rfeatures)])

predY = []
for n in range(0,len(y)):
    pY = regr.predict(cX[n,:].reshape((1,len(Rfeatures))))
    predY.extend(pY)

modelDf = pd.DataFrame({'predY':predY, 'y':y}) 
                     
with sns.plotting_context("notebook",font_scale=1.2):
    sns.jointplot(x="y", y="predY", data = modelDf, kind="reg")
  
  
#save regression models as follows  
regr_ACT_level2 = regr;
joblib.dump(regr_ACT_level2, 'regressionModelLevel2.pkl')
pickle.dump(regr_ACT_level2, open( "regr_ACT_level2.p", "wb" ))

#Record the end accuracy and save it as an object using pickle
GroupEndAccuracy = df.EndAccuracy

#ax = plt.figure()
#with sns.plotting_context("notebook",font_scale=1.2):
#     ax = sns.distplot(GroupEndAccuracy, hist=False, rug=True, color='blue');
#     ax = sns.kdeplot(GroupEndAccuracy, shade=True, color='blue');
#ax.set(yticks=[])
#ax.legend_.remove()
#ax.set_xlabel('End Accuracy', fontsize=16)
#ax.hold(True)

pickle.dump(GroupEndAccuracy, open( "GroupEndAccuracyLevel2.p", "wb" ))



#%% Below is the same function integrated in the code above to smoothe accuracies,
# find the accurac plateau value and index and plot the graphs for each patient 
# to better understand smoothing effects 
 
def zero_slope(data, chunksize, max_slope = .001):
    midindex = chunksize / 2
    for index in xrange(len(data)/chunksize):
        start = (index * chunksize)
        chunk = data[start + 1 : start + chunksize, :]
        dx, dy = abs(chunk[0] - chunk[-1])
        #print dy, dx, dy / dx
        if 0 <= dy/dx < max_slope:
            return chunk[midindex]
    return []        

A = []
NO_DATA = np.array([0, 0])
BAD_DATA = np.array([1, 1])  
for n in range(0,len(responses)):
    
    y_raw = responses[n].accuracies
    
    if len(y_raw) < 33:
        A.append(NO_DATA)
        #print len(y_raw)
        continue
    
    win_len = int(.1*len(y_raw))
    if win_len<9:
       win_len=9
       
    y_s = pd.rolling_mean(np.array(y_raw[3:]), win_len+1)
    ydata = y_s[win_len:]
    xdata = range(0,len(ydata))
    z = np.polyfit(xdata, ydata, 4)
    f2  = np.poly1d(z)
    x = np.linspace(np.min(xdata), np.max(xdata), len(xdata)*8)
    ynew = f2(x)
    
    
#    fig = plt.figure(figsize=(10,4.5))
#    fig.suptitle('Response Accuracy', fontsize=14, fontweight='bold')
#    fig.subplots_adjust(top=0.85)
#    plt.subplot(121) 
#    plt.plot(y_raw, '.-'), plt.ylim([0,1.05])
#    plt.xlabel('trials',fontsize=14)
#    plt.ylabel('accuracy',fontsize=14)
#    plt.title('raw accuracies for one patient', fontsize=14)
#    plt.subplot(122) 
#    plt.plot(xdata, ydata, '.-', label='moving average smoothing'), plt.ylim([0,1.05])
#    plt.plot(x,ynew, label='polynomial fit')
#    plt.legend(loc='best', prop={'size':12})
#    plt.xlabel('trials',fontsize=14)
#    plt.ylabel('accuracy',fontsize=14)
#    plt.title('smoothened accuracies', fontsize=14)
#    plt.show()
   
    arrayList = list()
    index = 0
    while index < len(x):
        arrayList.append((x[index], ynew[index]))
        index += 1
    data = np.array(arrayList)
    chunksize = int(len(data)/20)
    val = zero_slope(data, chunksize, max_slope = .001)
    print(val)
    
    if len(val) > 0:        
       A.append(val)
    else:
       A.append(BAD_DATA)
       print n
    
#shape(np.where(A==BAD_DATA))

#plot group results
accuracyIndPlateau = [an[0] for an in A]
accuracyValPlateau = [an[1] for an in A] 

py.figure()
ax = plt.subplot(121, xlabel='task # at plateau', ylabel='frequency', title='Plateau Index for Accuracy') 
py.hist(accuracyIndPlateau, 30)
ax = plt.subplot(122, xlabel='accuracy at plateau', ylabel='frequency', title='Plateau Accuracy') 
py.hist(accuracyValPlateau, 30) 
py.legend(loc='best')
py.show()



#%% Below is the same function integrated in the code above to smooth latencies
# find the minimum value and index, and plot the graphs for each patient 
# to better understand smoothing effects 

B = []
NO_DATA = np.array([0, 0])
BAD_DATA = np.array([1, 1])  
for n in range(0,len(responses)):
    
    y_raw = responses[n].latencies
    
    if len(y_raw) < 33:
        B.append(NO_DATA)
        #print len(y_raw)
        continue
    
    win_len = int(.1*len(y_raw))
    if win_len<9:
       win_len=9
       
    y_s = pd.rolling_mean(np.array(y_raw[3:]), win_len+1)
    ydata = y_s[win_len:]
    xdata = range(0,len(ydata))
    z = np.polyfit(xdata, ydata, 8)
    f2  = np.poly1d(z)
    x = np.linspace(np.min(xdata), np.max(xdata), len(xdata)*2)
    ynew = f2(x)
    
#    fig = plt.figure(figsize=(10,4.5))
#    fig.suptitle('Response Latency', fontsize=14, fontweight='bold')
#    fig.subplots_adjust(top=0.85)
#    plt.subplot(121) 
#    plt.plot(y_raw, 'o-', color='g'), plt.ylim([0,int(np.max(y_raw))+5])
#    plt.xlabel('trials',fontsize=14)
#    plt.ylabel('latency (s)',fontsize=14)
#    plt.title('raw latency for one patient', fontsize=14)
#    plt.subplot(122) 
#    plt.plot(xdata, ydata, 'o-', color='g', label='moving average smoothing')
#    plt.plot(x,ynew, label='polynomial fit'), plt.ylim([30,int(np.max(ydata))+5])
#    plt.legend(loc='best', prop={'size':12})
#    plt.xlabel('trials',fontsize=14)
#    plt.ylabel('latency (s)',fontsize=14)
#    plt.title('smoothened latency', fontsize=14)
#    plt.show()
        
    value = np.min(ynew)    
    location = np.where(ynew == value)
    val = [location[0][0]/2, value]
    print(val)
    
    if len(val) > 0:        
       B.append(val)
    else:
       B.append(BAD_DATA)
       print n
    
#shape(np.where(B==BAD_DATA))



