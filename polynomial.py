import matplotlib
import matplotlib.patheffects
import numpy
import csv
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
  
FILENAME = "nitrate.csv"
FILEPATH = "nutrients/cluster1/"
DATACOL = 9
STATION = "I"
DEGREE = 3
YEARFIRST = 2009
YEARLAST = 2021
YLABEL = "NH4 (Micromolars)"
DOTCOLOR = "magenta"
LINECOLOR = "purple"
LINETYPE = "solid"
MARK = "d"
LEFTSCALE = numpy.arange(0, 2.1, .5)
BOTTOMSCALE = numpy.arange(2008, 2023, 2)
CFACTOR = 1.0

with open(FILEPATH + FILENAME, encoding='utf-8-sig') as file:
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
            
    x = numpy.array(xlist).reshape(-1, 1)
    y = numpy.array(ylist).reshape(-1, 1)
    poly = PolynomialFeatures(degree=DEGREE, include_bias=False)
    curvex = poly.fit_transform(x)
    curvex = sm.add_constant(curvex)
    model = sm.OLS(y, curvex).fit(method = 'qr')
    curvey = model.predict(curvex)
        
    nutrient = (FILENAME.split('.'))[0]
    nutrient = nutrient[0].upper() + nutrient[1:]
    
    print(model.summary())
    
    rs = model.rsquared_adj
    p = model.f_pvalue
    fig, axes = matplotlib.pyplot.subplots()
    fig.text(.1,.89, f"R\u00b2 = {rs: .5f}     P = {p: .5f}     Degree = {DEGREE}", color = DOTCOLOR).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
    axes.scatter(x, y, color=DOTCOLOR, marker = MARK, label = nutrient)
    axes.plot(x,curvey, marker = "", linestyle = LINETYPE, color=LINECOLOR, label = nutrient + " Trend")
    axes.set_yticks(LEFTSCALE)
    axes.set_xticks(BOTTOMSCALE)
    matplotlib.pyplot.xlabel("Year")
    matplotlib.pyplot.ylabel(YLABEL, color=DOTCOLOR).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
    
    nutrient = (FILENAME.split('.'))[0]
    nutrient = nutrient[0].upper() + nutrient[1:]
    title = nutrient + " (" + (str)(years[0]) + " - " + (str)(years[-1]) + ") Station " + STATION
    matplotlib.pyplot.xlabel("Year")
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.ylabel(YLABEL)
    fig.text(.05,.75, STATION, fontsize=30, ha="center")
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.tight_layout(pad=4)
    fig.legend(ncol=4, loc = "upper left")
    matplotlib.pyplot.show()