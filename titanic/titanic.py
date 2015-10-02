import csv
import numpy as np
csv_file_object = csv.reader(open('train.csv', 'rb'))
csv_file_object.next()
data=[]
for row in csv_file_object:
	data.append(row)
data=np.array(data)
number_passengers=np.size(data[0::,1].astype(np.float))
number_survived=np.size(data[0::,1].astype(np.float))
women_only=data[0::,4]=='female'
men_only=data[0::,4]!='female'
women_float=data[women_only,1].astype(np.float)
men_float=data[men_only,1].astype(np.float)
men_survive=np.sum(men_float)/np.size(men_float)
women_survive=np.sum(women_float)/np.size(women_float)
print men_survive
print women_survive
		
