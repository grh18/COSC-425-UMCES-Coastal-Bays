import matplotlib
import matplotlib.patheffects
import numpy
import csv
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

FILENAME = "phosphate.csv"
FILENAME2 = "silicate.csv"
FILEPATH = "nutrients/cluster1/"
DATACOL = 9
STATION = "I"
YEARFIRST = 1995
YEARLAST = 2008
YLABEL = "PO4 (Micromolars)"
YLABEL2 = "SI(OH)4 (Micromolars)"
DOTCOLOR = "orange"
LINECOLOR = "red"
LINETYPE = "solid"
MARK = "d"
DOTCOLOR2 = "cyan"
LINECOLOR2 = "blue"
LINETYPE2 = "dotted"
MARK2 = "s"
LEFTSCALE = numpy.arange(0, .81, .2)
RIGHTSCALE = numpy.arange(0, 49, 12)
BOTTOMSCALE = numpy.arange(1994, 2011, 2)
CFACTOR = 3.0
CFACTOR2 = 1.0

def plot (file, cfactor):
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
    for row in data:
        if ((int)(row[2]) < YEARFIRST):
            continue
        if ((int)(row[2]) > YEARLAST):
            break;
        if (row[DATACOL] == ''):
            continue
        if ((int)(row[2]) == curyear):
            yearsum += (float)(row[DATACOL]) / cfactor
            samplenum += 1
        else:
            if (samplenum != 0):
                ylist.append(yearsum / samplenum)
                xlist.append(((int)(row[2])))
            samplenum = 1
            yearsum = (float)(row[DATACOL]) / cfactor
            curyear += 1
        
        if ((int)(row[2]) not in years):
            years.append((int)(row[2]))
    
    x = numpy.array(xlist)
    y = numpy.array(ylist)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=123)
    slope, intercept, r, p, std_err = scipy.stats.linregress(x_train, y_train)
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    def yline(x):
        return slope * x + intercept
    y_pred = list(map(yline, x_test))
    
    return  x, y, x_test, y_pred, r, p, years
    
def printstats (nutrient, r, p):
    print("\n" + nutrient)
    print("-" * 25)
    print(f"R\u00b2:       {r * r: .5f}")
    print(f"P:         {p:.5f}")
    print("\n")

fig, axes = matplotlib.pyplot.subplots()

with open(FILEPATH + FILENAME, encoding='utf-8-sig') as file:
    x, y, x_test, y_pred, r, p, years = plot(file, CFACTOR)
    
    nutrient = (FILENAME.split('.'))[0]
    nutrient = nutrient[0].upper() + nutrient[1:]
    printstats(nutrient, r, p)
    
    title = nutrient + " (" + (str)(years[0]) + " - " + (str)(years[-1]) + ") Station " + STATION
    
    fig.text(.1,.89, f"P = {p:.5f}    R\u00b2 = {r * r: .5f}", color = DOTCOLOR).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
    axes.scatter(x, y, color=DOTCOLOR, marker = MARK, label = nutrient)
    axes.plot(x_test,y_pred, marker = "", linestyle = LINETYPE, color=LINECOLOR, label = nutrient + " Trend")
    axes.set_yticks(LEFTSCALE)
    axes.set_xticks(BOTTOMSCALE)
    matplotlib.pyplot.xlabel("Year")
    matplotlib.pyplot.ylabel(YLABEL, color=DOTCOLOR).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
    
if (FILENAME2 != ""):
    with open(FILEPATH + FILENAME2, encoding='utf-8-sig') as file2:
        x, y, x_test, y_pred, r, p, years = plot(file2, CFACTOR2)
        
        nutrient2 = (FILENAME2.split('.'))[0]
        nutrient2 = nutrient2[0].upper() + nutrient2[1:]
        printstats(nutrient2, r, p)
        
        title = nutrient + " & " + nutrient2 + " (" + (str)(years[0]) + " - " + (str)(years[-1]) + ") Station " + STATION
        
        fig.text(.6,.89, f"P = {p:.5f}    R\u00b2 = {r * r: .5f}", color = DOTCOLOR2).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
        axes2 = axes.twinx()
        axes2.scatter(x, y, color=DOTCOLOR2, marker = MARK2, label = nutrient2)
        axes2.plot(x_test,y_pred, marker = "", linestyle = LINETYPE2, color=LINECOLOR2, label=nutrient2 + " Trend")
        axes2.set_yticks(RIGHTSCALE)
        axes2.set_xticks(BOTTOMSCALE)
        matplotlib.pyplot.ylabel(YLABEL2, color= DOTCOLOR2).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])

fig.text(.05,.75, STATION, fontsize=30, ha="center")
matplotlib.pyplot.title(title)
matplotlib.pyplot.tight_layout(pad=4)
fig.legend(ncol=4, loc = "upper left")
matplotlib.pyplot.show()
            
            
            