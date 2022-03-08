import csv
import numpy as n




# Basically, I create a matrix X which is simply all the columns excluding the first two
# I also create a matrix called y_name which is an array of each label in every row

def convert(name):
    with open(name) as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        for i,row in enumerate(reader):
            if(i > 0):
                if(i == 1):
                    X = n.matrix(row[2:])
                    y_name = n.array(row[1])
                else:

                    y_name = n.append(y_name,[row[1]])
                    X = n.append(X,[row[2:]],axis = 0)

        return X, y_name