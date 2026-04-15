import matplotlib
import matplotlib.patheffects
import numpy
import scipy
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import csv

FILENAME = "phosphate.csv"
FILEPATH = "nutrients/Cluster1/"
DATACOL = 9
YEARCOL = 2
STATION = "I"
YEARFIRST = 1995
YEARLAST = 2022
YLABEL = "micrograms per liter"
DOTCOLOR = "orange"
LINECOLOR = "red"
LINETYPE = "solid"
LINECOLOR2 = "magenta"
LINETYPE2 = "dotted"
MARK = "d"
LEFTSCALE = numpy.arange(0, 1.1, .25)
BOTTOMSCALE = numpy.arange(1994, 2025, 2)
CFACTOR = 3
DEGREE = 5

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
        if ((int)(row[YEARCOL]) < YEARFIRST):
            continue
        if ((int)(row[YEARCOL]) > YEARLAST):
            break;
        if (row[DATACOL] == ''):
            continue
        if ((int)(row[YEARCOL]) == curyear):
            yearsum += (float)(row[DATACOL].replace(",", "")) / cfactor
            samplenum += 1
        else:
            if (samplenum != 0):
                ylist.append(yearsum / samplenum)
                xlist.append(((float)(row[YEARCOL])))
            samplenum = 1
            yearsum = (float)(row[DATACOL].replace(",", "")) / cfactor
            curyear += 1
        
        if ((int)(row[YEARCOL]) not in years):
            years.append((int)(row[YEARCOL]))
        
    x = numpy.array(xlist)
    y = numpy.array(ylist)
    
    slope, intercept, r, p, std_err = scipy.stats.linregress(x, y)
    
    def yline(x):
        return slope * x + intercept
    linreg = list(map(yline, x))
    
    return  x, y, r, p, std_err, linreg, years

fig, axes = matplotlib.pyplot.subplots()

with open(FILEPATH + FILENAME, encoding='utf-8-sig') as file:
    x, y, r, p, std_err, linreg, years = plot(file, CFACTOR)
    
    nutrient = (FILENAME.split('.'))[0]
    nutrient = nutrient[0].upper() + nutrient[1:]
    
    title = nutrient + " (" + (str)(years[0]) + " - " + (str)(years[-1]) + ") Station " + STATION 
    
    fig.text(.1,.89, f"P = {p:.5f}    R\u00b2 = {r*r: .5f}", color = LINECOLOR).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
    axes.scatter(x, y, color=DOTCOLOR, marker = MARK)
    axes.plot(x, linreg, marker = "", linestyle = LINETYPE, color=LINECOLOR, label = "Linear Trend")
    matplotlib.pyplot.xlabel("Year")
    matplotlib.pyplot.ylabel(YLABEL, color=DOTCOLOR).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
    x = numpy.array(x).reshape(-1, 1)
    y = numpy.array(y).reshape(-1, 1)
    poly = PolynomialFeatures(degree=DEGREE, include_bias=True)
    curvex = poly.fit_transform(x)
    curvex = sm.add_constant(curvex)
    model = sm.OLS(y, curvex).fit(method = 'qr')
    curvey = model.predict(curvex)
    axes.plot(x, curvey, marker = "", linestyle = LINETYPE2, color = LINECOLOR2, label = f"Degree {DEGREE} Polynomial Trend")       
    axes.set_yticks(LEFTSCALE)
    axes.set_xticks(BOTTOMSCALE)
    matplotlib.pyplot.xticks(rotation=60)
    rsq = model.rsquared_adj
    pval = model.f_pvalue
    fig.text(.6,.89, f"P = {pval:.5f}    R\u00b2 = {rsq: .5f}", color = LINECOLOR2).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])

fig.text(.05,.75, STATION, fontsize=30, ha="center")
matplotlib.pyplot.title(title)
matplotlib.pyplot.tight_layout(pad=5)
fig.legend(ncol=2, loc = "upper center")
matplotlib.pyplot.show()