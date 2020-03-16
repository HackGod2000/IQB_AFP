import pandas as pd
import numpy as np 
from sklearn import svm # import for SVM i.e. Machine Learning Model 

train_data = pd.read_csv("train.csv") 
test_data = pd.read_csv("test.csv")
arr=[] # train Array 
arr_t = [] # test Array


'''Feature 1 : This part deals with generating the binary profile for the entire sequence 
    it generates an array of size 20(sub_arr) corressponding to each sequence which is then stored in the final array (arr)
    this array is further passes as a feature of the SVM model later in the Code'''
for i in train_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}


    keys= amino.keys()
    for j in i:
        if j in keys:
            amino[j] +=1

    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr.append(sub_arr)

#print(arr[0])
for i in test_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}

    keys= amino.keys()
    for j in i:
        if j in keys:
            amino[j] +=1


    sub_arr=[]
    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr_t.append(sub_arr)
#------------------------------------------------------------------------------------------------------------------------------------------------#
#print(arr)

Sequences=['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']
arr1=[]
arr1_t= []

for i in train_data["Sequence"]:

    pep_s= {}
    for _ in Sequences:
        for w in Sequences:
            dipep = _+w
            pep_s[dipep]=0   

    keys = pep_s.keys()     
    for j in range(0,len(i)-1 ):
        temp_pep = i[j]+i[j+1]
        if temp_pep in keys:
            pep_s[temp_pep]+=1

    sub_arr=[]
    for k in pep_s:
        x=pep_s[k]
        sub_arr.append(x)
    arr1.append(sub_arr)



for i in test_data["Sequence"]:

    pep_s= {}
    for _ in Sequences:
        for w in Sequences:
            dipep = _+w
            pep_s[dipep]=0   

    keys = pep_s.keys()     
    for j in range(0,len(i)-1 ):
        temp_pep = i[j]+i[j+1]
        if temp_pep in keys:
            pep_s[temp_pep]+=1

    sub_arr=[]
    for k in pep_s:
        x=pep_s[k]
        sub_arr.append(x)
    arr1_t.append(sub_arr)
#-----------------------------------------------------------------------------------------------------------------------------------#




train_data["b_profile"]=arr
test_data["b_profile"]=arr_t
train_data["dipep"]=arr1
test_data["dipep"]=arr1_t

#print(train_data.dtypes)
#print(arr1+arr)

feature_df = train_data['b_profile']+train_data['dipep']
feature_dft = test_data['b_profile']+test_data['dipep']
# print(feature_df)
x_train= list(feature_df) #independent Variable
y_train= np.asarray(train_data['Lable'])# Dependent Variable
x_test = list(feature_dft)
#print(x_train)
print(y_train)


classifier = svm.SVC()
classifier.fit(x_train,train_data['Lable'])
y_predict = classifier.predict(x_test)
print(y_predict)
output = pd.DataFrame({'ID':test_data.ID , 'Label':y_predict})
output.to_csv('ans2.csv', index = False)

print("Sub success")