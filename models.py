import numpy
import csv
import statsmodels.api as sm
import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

def getData (file, yearfirst, yearlast, datacol, yearcol, cfactor):
    data=[]
    reader = csv.reader(file)
    for row in reader:
        data.append(row)
        
    xlist = []
    ylist = []
    years = []
    yearsum = 0.0
    samplenum = 1.0
    curyear = yearfirst
    for row in data:
        if ((int)(row[yearcol]) < yearfirst):
            continue
        if ((int)(row[yearcol]) > yearlast):
            break;
        if (row[datacol] == ''):
            continue
        if ((int)(row[yearcol]) == curyear):
            yearsum += (float)(row[datacol].replace(",", "")) / cfactor
            samplenum += 1
        else:
            if (samplenum != 0):
                ylist.append(yearsum / samplenum)
                xlist.append(((float)(row[yearcol])))
            samplenum = 1
            yearsum = (float)(row[datacol].replace(",", "")) / cfactor
            curyear += 1
        
        if ((int)(row[yearcol]) not in years):
            years.append((int)(row[yearcol]))
    
    return  xlist, ylist, years

def linear (xlist, ylist, linecolor, linetype, axes):
    x = numpy.array(xlist)
    y = numpy.array(ylist)
    
    slope, intercept, r, p, std_err = scipy.stats.linregress(x, y)
    
    def yline(x):
        return slope * x + intercept
    linreg = list(map(yline, x))
    
    axes.plot(x, linreg, marker = "", linestyle = linetype, color = linecolor, label = "Basic Linear Regression")
    
    return linreg, r * r, p

def splitLinear (xlist, ylist, linecolor, linetype, axes):
    x = numpy.array(xlist)
    y = numpy.array(ylist)
    
    x_train, x_test = TimeSeriesSplit(n_splits=3, test_size=2, gap=2).split(x)
    slope, intercept, r, p, std_err = scipy.stats.linregress(x_train, y)
    model = LinearRegression()
    model.fit(x_train, y)
    
    def yline(x):
        return slope * x + intercept
    y_pred = list(map(yline, x_test))
    
    axes.plot(x_test,y_pred, marker = "", linestyle = linetype, color = linecolor, label = "Linear Regression with Train / Test Split")
    
    return x_test, y_pred, r, p

def polynomial (degree, xlist, ylist, linecolor, linetype, axes):
    x = numpy.array(xlist).reshape(-1, 1)
    y = numpy.array(ylist).reshape(-1, 1)
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    curvex = poly.fit_transform(x)
    curvex = sm.add_constant(curvex)
    model = sm.OLS(y, curvex).fit(method = 'qr')
    curvey = model.predict(curvex)
    
    axes.plot(x, curvey, marker = "", linestyle = linetype, color = linecolor, label = "Polynomial Regression")
    
    return curvey, model.rsquared_adj, model.f_pvalue