import matplotlib
import matplotlib.pyplot
import matplotlib.patheffects
import numpy
import scipy
import csv                           # Included Libraries

FILENAME = "phosphate.csv"                 # Primary data file
FILENAME2 = "silicate.csv"                     # Secondary data file ("" if n/a.)
FILEPATH = ""
DATACOL = 5                          # Column index in the CSV that holds the measurement
STATION = "VII"                         # Station/Cluster Number where data is collected from.

# Range in years to include in analysis.
YEARFIRST = 1991
YEARLAST = 2008

# Label and sytling for primary dataset. (Left Y-axis)
YLABEL = "Phosphate (Micromolar)"
DOTCOLOR = "blue"
LINECOLOR = "blue"
LINETYPE = "solid"
MARK = "D"
FILLSTYLE1 = 'full'

# Label and styling for secondary dataset. (Right Y-axis, only runs if there is secondary dataset.)
YLABEL2 = "Silicate (Micromolar)"
DOTCOLOR2 = "green"
LINECOLOR2 = "green"
LINETYPE2 = "dashed"
MARK2 = "s"
FILLSTYLE2 = 'none' #leave as 'full' if no specific marker fillstyle (default is full)

# Scaling and Conversion Factors
LEFTSCALE = numpy.arange(0, 2, .4)        # Left Y-axis scale
RIGHTSCALE = numpy.arange(0, 120, 24)        # Right Y-axis scale
BOTTOMSCALE = numpy.arange(1992, 2012, 2)   # X-axis scale
CFACTOR = 1.0
CFACTOR2 = 1.0

def plot (file, cfactor):
    data=[]
    reader = csv.reader(file)
    for row in reader:
        data.append(row)
        
    xlist = []               
    ylist = []               
    years = []              
    
    yearsum = 0.0           # Cumulative Sum within the current value and all preceding values, commonly used to current year.
    samplenum = 0.0         # Number of samples within the current year.

    curyear = YEARFIRST     
    for row in data: 
        if ((int)(row[2]) < YEARFIRST):     # Any rows before the beginning year is skipped. 
            continue
        if ((int)(row[2]) > YEARLAST):      # End of program once the last year is passed.
            break;
        if (row[DATACOL].strip() == ''):    # Skips rows where data entry is empty or whitespace.
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
            curyear = (int)(row[2])   
            # curyear += 1
        
        if ((int)(row[2]) not in years):
            years.append((int)(row[2]))
        
    x = numpy.array(xlist)
    y = numpy.array(ylist)
    
    slope, intercept, r, p, std_err = scipy.stats.linregress(x, y)
    
    def yline(x):
        return slope * x + intercept
    linreg = list(map(yline, x))
    
    return  x, y, r, p, std_err, linreg, years
    
def printstats (pigment, r, p, std_err):
    print("\n" + pigment)
    print("-" * 25)
    print(f"R\u00b2:       {r*r: .5f}")
    print(f"P:         {p:.5f}")
    print(f"Std Err:   {std_err:.5f}")
    print("\n")

fig, axes = matplotlib.pyplot.subplots()

with open(FILEPATH + FILENAME, encoding='utf-8-sig') as file:
    x, y, r, p, std_err, linreg, years = plot(file, CFACTOR)
    
    pigment = (FILENAME.split('.'))[0]
    pigment = pigment[0].upper() + pigment[1:]
    printstats(pigment, r, p, std_err)
    
    title = pigment + " (" + (str)(years[0]) + " - " + (str)(years[-1]) + ") Station " + STATION
    
    fig.text(.1,.89, f"P = {p:.5f}    R\u00b2 = {r*r: .5f}", color = DOTCOLOR).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
    axes.scatter(x, y, edgecolors=DOTCOLOR, facecolors=DOTCOLOR if FILLSTYLE1 != 'none' else 'none', marker = MARK, label = pigment)
    axes.plot(x,linreg, marker = "", linestyle = LINETYPE, color=LINECOLOR, label = pigment + " Trend")
    axes.set_yticks(LEFTSCALE)
    axes.set_ylim(LEFTSCALE[0], LEFTSCALE[-1])
    axes.set_xticks(BOTTOMSCALE)
    matplotlib.pyplot.xlabel("Year")
    matplotlib.pyplot.ylabel(YLABEL, color=DOTCOLOR).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
    
if (FILENAME2 != ""):
    with open(FILEPATH + FILENAME2, encoding='utf-8-sig') as file2:
        x, y, r, p, std_err, linreg, years = plot(file2, CFACTOR2)
        
        pigment2 = (FILENAME2.split('.'))[0]
        pigment2 = pigment2[0].upper() + pigment2[1:]
        printstats(pigment2, r, p, std_err)
        
        title = pigment + " & " + pigment2 + " (" + (str)(years[0]) + " - " + (str)(years[-1]) + ") Station " + STATION
        
        fig.text(.6,.89, f"P = {p:.5f}    R\u00b2 = {r*r: .5f}", color = DOTCOLOR2).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])
        axes2 = axes.twinx()
        axes2.scatter(x, y, edgecolors=DOTCOLOR2, facecolors=DOTCOLOR2 if FILLSTYLE2 != 'none' else 'none', marker = MARK2, label = pigment2)
        axes2.plot(x,linreg, marker = "", linestyle = LINETYPE2, color=LINECOLOR2, label=pigment2 + " Trend")
        axes2.set_yticks(RIGHTSCALE)
        axes2.set_ylim(RIGHTSCALE[0], RIGHTSCALE[-1])
        axes2.set_xticks(BOTTOMSCALE)
        # axes.scatter(x, y, color=DOTCOLOR2, marker=MARK2, label=pigment2)
        # axes.plot(x, linreg, marker="", linestyle=LINETYPE2, color=LINECOLOR2, label=pigment2 + " Trend")
        # matplotlib.pyplot.ylabel(YLABEL, color=DOTCOLOR).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])

        matplotlib.pyplot.ylabel(YLABEL2, color= DOTCOLOR2).set_path_effects([matplotlib.patheffects.withSimplePatchShadow(offset=(.6, -.6), shadow_rgbFace="black", alpha = .5, rho = 0)])

fig.text(.05,.75, STATION, fontsize=30, ha="center")
matplotlib.pyplot.title(title)
matplotlib.pyplot.tight_layout(pad=4)
fig.legend(ncol=4, loc = "upper left")
matplotlib.pyplot.show()
    
            