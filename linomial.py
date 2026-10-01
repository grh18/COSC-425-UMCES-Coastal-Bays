import matplotlib
import matplotlib.patheffects
import numpy
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import PolynomialFeatures
import csv

FILENAME = "chlorophyll.csv"
FILEPATH = "nutrients/Cluster7/"
NAME = "Chlorophyll a"
YorM = "y"
DATACOL = 9
YEARCOL = 2
MONTHCOL = 4
SEGMENT = "VII"
YEARFIRST = 1995
YEARLAST = 2021
YLABEL = "μg / L"
DOTCOLOR = "green"
LINECOLOR = "red"
LINETYPE = "solid"
LINECOLOR2 = "magenta"
LINETYPE2 = "solid"
MARK = "d"
LEFTSCALE = numpy.arange(0, 10.5, 2.5)
BOTTOMSCALE = numpy.arange(1995, 2023, 2)
CFACTOR = 1.0
DEGREE = 4
#WIDTH = (YEARLAST - YEARFIRST) / 3

def plot (file, cfactor):
    data=[]
    reader = csv.reader(file)
    for row in reader:
        data.append(row)
        
    xlist = []
    ylist = []
    xlistm = []
    ylistm = []
    years = []
    yearsum = 0.0
    monthsum = 0.0
    samplenum = 1.0
    msamplenum = 1.0
    curyear = YEARFIRST
    curmonth = 1
    for row in data:
        if ((int)(row[YEARCOL]) < YEARFIRST):
            continue
        if ((int)(row[YEARCOL]) > YEARLAST):
            break;
        if (row[DATACOL] == ''):
            continue
        if (row[MONTHCOL] == curmonth):
            monthsum += (float)(row[DATACOL].replace(",", "")) / cfactor
            msamplenum += 1
        else: 
            if (msamplenum != 0):
                ylistm.append(monthsum / msamplenum)
                xlistm.append((int(row[YEARCOL]) + ((int)(curmonth) / 12.0)))
            msamplenum = 1
            monthsum = (float)(row[DATACOL].replace(",", "")) / cfactor
            curmonth = row[MONTHCOL]
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
    ym = numpy.array(ylistm)
    xm = numpy.array(xlistm)
    
    return  x, y, xm, ym, years

fig, axes = matplotlib.pyplot.subplots()

with open(FILEPATH + FILENAME, encoding='utf-8-sig') as file:
    x, y, xm, ym, years = plot(file, CFACTOR)
    if (YorM == "y" or YorM == "Y"):
        axes.scatter(x, y, color=DOTCOLOR, marker = MARK)
        xc = sm.add_constant(x)
        model = sm.OLS(y, xc).fit(method="qr")
    else:
        axes.scatter(xm, ym, color=DOTCOLOR, marker = MARK)
        xc = sm.add_constant(xm)
        model = sm.OLS(ym, xc).fit(method="qr")
        x = xm
        y = ym
    lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(model.resid_pearson, xc)
    print(lm_pvalue)
    if (lm_pvalue >= .06):
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
        curvey = model.predict(curvex)
        rsq = model.rsquared
        pval = model.f_pvalue
        axes.plot(x, curvey, marker = "", linestyle = LINETYPE, color = LINECOLOR, label = "WLS Linear trend")
        fig.text(0.2,.89, f"P = {pval:.5f}", color = LINECOLOR).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
        fig.text(0.2,.85, f"R\u00b2 = {rsq: .5f}", color = LINECOLOR).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
        
    x2 = numpy.array(x).reshape(-1, 1)
    y2 = numpy.array(y).reshape(-1, 1)
    poly = PolynomialFeatures(degree=DEGREE, include_bias=True)
    curvex = poly.fit_transform(x2)
    curvex = sm.add_constant(curvex)
    if (lm_pvalue >= .06):
        model = sm.OLS(y2, curvex).fit(method = 'qr')
        curvey = model.predict(curvex)
        axes.plot(x, curvey, marker = "", linestyle = LINETYPE2, color = LINECOLOR2, label = f"Degree {DEGREE} Polynomial Trend")   
        rsq = model.rsquared
        pval = model.f_pvalue
        fig.text(.5,.89, f"P = {pval:.5f} ", color = LINECOLOR2).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])    
        fig.text(.5,.85, f"R\u00b2 = {rsq: .5f}", color = LINECOLOR2).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
        
    else:
        w = sm.OLS(y, curvex).fit(method="qr").resid_pearson
        model = sm.WLS(y2, curvex, weights=1.0 / w**2).fit(method = 'qr')
        curvey = model.predict(curvex)
        axes.plot(x, curvey, marker = "", linestyle = LINETYPE2, color = LINECOLOR2, label = f"Degree {DEGREE} Polynomial Trend")   
        rsq = model.rsquared
        pval = model.f_pvalue
        fig.text(.5,.89, f"P = {pval:.5f} ", color = LINECOLOR2).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])    
        fig.text(.5,.85, f"R\u00b2 = {rsq: .5f}", color = LINECOLOR2).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])

    nutrient = (FILENAME.split('.'))[0]
    nutrient = nutrient[0].upper() + nutrient[1:]
    nutrient = NAME
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
#matplotlib.pyplot.gcf().set_figwidth(WIDTH) 
matplotlib.pyplot.show()