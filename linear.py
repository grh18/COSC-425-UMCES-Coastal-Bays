import matplotlib
import numpy
import scipy
import csv

FILENAME = "phosphate.csv"
FILENAME2 = ""
YEARFIRST = 2009
YEARLAST = 2022
YLABEL = "PO4 (Micromolars)"
YLABEL2 = "NO3 (Micromolars)"
DOTCOLOR = "orange"
LINECOLOR = "red"
LINETYPE = "solid"
MARK = "d"
DOTCOLOR2 = "blue"
LINECOLOR2 = "darkblue"
LINETYPE2 = "dotted"
MARK2 = "s"
LEFTSCALE = numpy.arange(0, 3.1, .5)
RIGHTSCALE = numpy.arange(0, 2.1, .5)
BOTTOMSCALE = numpy.arange(2009, 2022, 2)
CFACTOR = 1.0
CFACTOR2 = 1.0

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
        
    x = numpy.array(xlist)
    y = numpy.array(ylist)
    
    slope, intercept, r, p, std_err = scipy.stats.linregress(x, y)
    
    nutrient = (FILENAME.split('.'))[0]
    nutrient = nutrient[0].upper() + nutrient[1:]
    nutrient2 = ""
    title=""
    if (FILENAME2 != ""):
        nutrient2 = (FILENAME2.split('.'))[0]
        nutrient2 = nutrient2[0].upper() + nutrient2[1:]
        title = nutrient + " & " + nutrient2 + " (" + (str)(years[0]) + " - " + (str)(years[-1]) + ")"
    else:
        title = nutrient + " (" + (str)(years[0]) + " - " + (str)(years[-1]) + ")"
    print("\n" + nutrient)
    print("-" * len(title))
    print("Slope: " + (str)(slope))
    print("Intercept: " + (str)(intercept))
    print("R-Squared: " + (str)(r * r))
    print("P-Value: " + (str)(p))
    print("Standard Error: " + (str)(std_err))
    print("\n")
    
    
    def yline(x):
        return slope * x + intercept
    linreg = list(map(yline, x))
    
    fig, axes = matplotlib.pyplot.subplots()
    axes.scatter(x, y, color=DOTCOLOR, marker = MARK, label = nutrient)
    axes.plot(x,linreg, marker = "", linestyle = LINETYPE, color=LINECOLOR)
    axes.set_yticks(LEFTSCALE)
    axes.set_xticks(BOTTOMSCALE)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.xlabel("Year")
    matplotlib.pyplot.ylabel(YLABEL)
    
    if (FILENAME2 != ""):
        with open(FILENAME2, encoding='utf-8-sig') as file2:
            data=[]
            reader2 = csv.reader(file2)
            for row in reader2:
                data.append(row)
                
            xlist = []
            ylist = []
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
                    yearsum += (float)(row[9]) / CFACTOR2
                    samplenum += 1
                else:
                    if (samplenum != 0):
                        ylist.append(yearsum / samplenum)
                        xlist.append((int)(row[2]))
                    samplenum = 1
                    yearsum = (float)(row[9]) / CFACTOR2
                    curyear += 1
                
                if ((int)(row[2]) not in years):
                    years.append((int)(row[2]))
                
            x = numpy.array(xlist)
            y = numpy.array(ylist)
            
            slope, intercept, r, p, std_err = scipy.stats.linregress(x, y)
            
            print("\n" + nutrient2)
            print("-" * len(title))
            print("Slope: " + (str)(slope))
            print("Intercept: " + (str)(intercept))
            print("R-Squared: " + (str)(r * r))
            print("P-Value: " + (str)(p))
            print("Standard Error: " + (str)(std_err))
            print("\n\n")
            
            def yline(x):
                return slope * x + intercept
            linreg = list(map(yline, x))
            axes2 = axes.twinx()
            axes2.scatter(x, y, color=DOTCOLOR2, marker = MARK2, label = nutrient2)
            axes2.plot(x,linreg, marker = "", linestyle = LINETYPE2, color=LINECOLOR2)
            axes2.set_yticks(RIGHTSCALE)
            axes2.set_xticks(BOTTOMSCALE)
            matplotlib.pyplot.ylabel(YLABEL2)
    matplotlib.pyplot.tight_layout(pad=3)
    fig.legend(loc = "upper left")
    matplotlib.pyplot.show()
            
            
            