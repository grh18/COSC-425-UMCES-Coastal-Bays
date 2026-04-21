import matplotlib
import matplotlib.patheffects
import numpy
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import PolynomialFeatures
import csv

FILENAME = "phosphate.csv"
FILEPATH = "nutrients/Cluster4/"
DATACOL = 9
YEARCOL = 2
SEGMENT = "IV"
YEARFIRST = 1995
YEARLAST = 2022
YLABEL = "micromolars"
DOTCOLOR = "orange"
LINECOLOR = "red"
LINETYPE = "solid"
LINECOLOR2 = "magenta"
LINETYPE2 = "solid"
MARK = "d"
LEFTSCALE = numpy.arange(0, 1.1, .2)
BOTTOMSCALE = numpy.arange(1994, 2023, 2)
CFACTOR = .75
DEGREE = 4

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
    
    return  x, y, years

fig, axes = matplotlib.pyplot.subplots()

with open(FILEPATH + FILENAME, encoding='utf-8-sig') as file:
    x, y, years = plot(file, CFACTOR)
    axes.scatter(x, y, color=DOTCOLOR, marker = MARK)
    
    xc = sm.add_constant(x)
    model = sm.OLS(y, xc).fit(method="qr")
    lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(model.resid_pearson, xc)
    print(lm_pvalue)
    if (lm_pvalue > .05):
        curvey = model.predict(xc)
        axes.plot(x, curvey, marker = "", linestyle = LINETYPE, color=LINECOLOR, label = "OLS Linear Trend")
        rsq = model.rsquared
        pval = model.f_pvalue
        fig.text(.2,.89, f"P = {pval:.5f}", color = LINECOLOR).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
        fig.text(.2,.85, f"R\u00b2 = {rsq: .5f}", color = LINECOLOR).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])  
    else:    
        curvex = sm.add_constant(x)
        w = sm.OLS(y, curvex).fit(method="qr").resid_pearson
        model = sm.WLS(y, curvex, weights = 1.0 / w**2).fit(method="qr")
        rsq = model.rsquared
        pval = model.f_pvalue
        curvey = model.predict(curvex)
        axes.plot(x, curvey, marker = "", linestyle = LINETYPE, color = LINECOLOR, label = "WLS Linear trend")
        fig.text(0.2,.89, f"P = {pval:.5f}", color = LINECOLOR).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
        fig.text(0.2,.85, f"R\u00b2 = {rsq: .5f}", color = LINECOLOR).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
        
    x2 = numpy.array(x).reshape(-1, 1)
    y2 = numpy.array(y).reshape(-1, 1)
    poly = PolynomialFeatures(degree=DEGREE, include_bias=True)
    curvex = poly.fit_transform(x2)
    curvex = sm.add_constant(curvex)
    model = sm.OLS(y2, curvex).fit(method = 'qr')
    curvey = model.predict(curvex)
    axes.plot(x, curvey, marker = "", linestyle = LINETYPE2, color = LINECOLOR2, label = f"Degree {DEGREE} Polynomial Trend")   
    rsq = model.rsquared
    pval = model.f_pvalue
    fig.text(.5,.89, f"P = {pval:.5f} ", color = LINECOLOR2).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])    
    fig.text(.5,.85, f"R\u00b2 = {rsq: .5f}", color = LINECOLOR2).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
    
    nutrient = (FILENAME.split('.'))[0]
    nutrient = nutrient[0].upper() + nutrient[1:]
    title = nutrient + " (" + (str)(years[0]) + " - " + (str)(years[-1]) + ") Segment " + SEGMENT 
    matplotlib.pyplot.xlabel("Year")
    matplotlib.pyplot.ylabel(YLABEL, color=DOTCOLOR).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
    axes.set_yticks(LEFTSCALE)
    axes.set_xticks(BOTTOMSCALE)
    matplotlib.pyplot.xticks(rotation=60)

fig.text(.05,.75, SEGMENT, fontsize=30, ha="center")
matplotlib.pyplot.title(title)
matplotlib.pyplot.tight_layout(pad=5)
fig.legend(ncol=3, loc = "upper center")
matplotlib.pyplot.show()