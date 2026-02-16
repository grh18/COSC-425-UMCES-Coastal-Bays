import matplotlib
import numpy
import scipy
import csv

with open("chlorophyl.csv", encoding='utf-8-sig') as file:
    data=[]
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

    xlist = []
    ylist = []
    for row in data[1:538]:
        print(row)
        if ( row[9] == '' ):
            continue
        ylist.append((float)(row[9]))     
        xlist.append((int)(row[2]))
    
    print(xlist)
    print(ylist)
    
    x = numpy.array(xlist)
    y = numpy.array(ylist)
    
    slope, intercept, r, p, std_err = scipy.stats.linregress(x, y)
    
    def ypoint(x):
      return slope * x + intercept
    
    linreg = list(map(ypoint, x))
    
    matplotlib.pyplot.scatter(x, y, color="blue")
    matplotlib.pyplot.plot(x, linreg, marker = "D", linestyle = "solid", color="darkblue")