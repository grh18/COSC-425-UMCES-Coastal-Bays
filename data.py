import matplotlib
import numpy
import scipy

def extractData(fileName):
    file = open(fileName, encoding='utf-8-sig')
    data = []
    colHeaders = file.readline().rstrip().split(',')
    data.append(colHeaders)
    file.seek(0)
    for line in file:
        data.append((file.readline().rstrip()).split(','))
    return data
 
data = extractData("ammonia.csv")

xlist = []
ylist = []
for row in data[1:]:
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

def point(x):
  return slope * x + intercept

linreg = list(map(point, x))

matplotlib.pyplot.scatter(x, y)
matplotlib.pyplot.plot(x, linreg)

#matplotlib.pyplot.plot(x, y, linestyle = "solid")
matplotlib.pyplot.show()