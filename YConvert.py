import numpy as n

# This is a function to convert the y_name matrix I created earlier to a y 
# matrix that uses integers to classify the labels instead of words

# What I essentially do here is I create a dictionary that assigns every new y_name to a number using count
# Then, for every value in the y_name array, I assign a new value equal to the number from y_dict
# So if E. Coli is E.Coli:4 in y_dict, it would be assigned 4 in the y matrix
def convertY(yNameList):
    
    #definitions:
    names = []
    count = 1
    for i,name in enumerate(yNameList):
        if(i == 0):
            names = [name]
            y_dict = {
                name:count
            }
            count += 1
            y = n.array(y_dict.get(name))
        else:
            if (name not in names):
                names.append(name)
                y_dict[name] = count
                count += 1
            y = n.column_stack((y,y_dict.get(name)))
    
    return y, len(y_dict)



    
                    

            