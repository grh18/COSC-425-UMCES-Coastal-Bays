import matplotlib
import numpy
import csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

FILENAME = "chlorophyll.csv"
YEARFIRST = 2009
YEARLAST = 2021
YLABEL = "Chlorophyll a (Grams per Litre)"
DOTCOLOR = "green"
LINECOLOR = "darkgreen"
LINETYPE = "solid"
MARK = "d"
LEFTSCALE = numpy.arange(0, 11, 2)
BOTTOMSCALE = numpy.arange(YEARFIRST,YEARLAST + 1, 2)
CFACTOR = 1

with open(FILENAME, encoding='utf-8-sig') as file:
    data=[]
    reader = csv.reader(file)
    for row in reader:
        data.append(row)
            
        xlist = []
        ylist = []
        years = []
        yearsum = 0.0
        samplenum = 1.0
        curyear = YEARFIRST
        for row in data[1:]:
            if ((int)(row[2]) < YEARFIRST):
                continue
            if ((int)(row[2]) > YEARLAST):
                break;
            if (row[9] == ''):
                continue
            if ((int)(row[2]) == curyear):
                yearsum += (float)(row[9]) / CFACTOR
                samplenum += 1
            else:
                if (samplenum != 0):
                    ylist.append(yearsum / samplenum)
                    xlist.append((int)(row[2]))
                samplenum = 1
                yearsum = (float)(row[9]) / CFACTOR
                curyear += 1
            
            if ((int)(row[2]) not in years):
                years.append((int)(row[2]))
            
    x = numpy.array(xlist).reshape(-1,1)
    y = numpy.array(ylist)
    
    pmodel = PolynomialFeatures(degree=3, include_bias=False)
    curvex = pmodel.fit_transform(x)
    lmodel = LinearRegression()
    lmodel.fit(curvex, y)
    curvey = lmodel.predict(curvex)
        
    nutrient = (FILENAME.split('.'))[0]
    nutrient = nutrient[0].upper() + nutrient[1:]
    print("\n" + nutrient)
    print("-" * len(nutrient))
    #print("Slope: " + (str)(slope))
    #print("Intercept: " + (str)(intercept))
    print("R-Squared: " + (str)(lmodel.score(curvex, y)))
    #print("Standard Error: " + (str)(std_err))
    print("\n")
        
    fig, axes = matplotlib.pyplot.subplots()
    matplotlib.pyplot.plot(x,curvey, marker = "", linestyle = LINETYPE, color=LINECOLOR, label = nutrient)
    matplotlib.pyplot.scatter(x, y, color=DOTCOLOR, marker = MARK)
    axes.set_yticks(LEFTSCALE)
    axes.set_xticks(BOTTOMSCALE)
    
    nutrient = (FILENAME.split('.'))[0]
    nutrient = nutrient[0].upper() + nutrient[1:]
    title = nutrient
    matplotlib.pyplot.xlabel("Year")
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.ylabel(YLABEL)
    matplotlib.pyplot.tight_layout(pad=3)
    matplotlib.pyplot.legend(loc = "upper left")
    matplotlib.pyplot.show()