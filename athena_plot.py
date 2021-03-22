#!/usr/bin/env python3

import argparse
import yaml
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants
from matplotlib.colors import LogNorm
from os import mkdir
from matplotlib import ticker
import athena_read
from math import floor,ceil

class Star:
  def __init__(self, problemFile, windno):
    self.mDot = float(problemFile["mdot"+str(windno)])
    self.vInf = float(problemFile["vinf"+str(windno)])
    self.avgMass = calculateAvgMass(problemFile,windno)

class CommonPlotting:
  def __init__(self,problemFile,config):
    if "xmin" not in config: self.xmin = problemFile["mesh"]["x1min"]
    else:                    self.xmin = config["xmin"]
    if "xmax" not in config: self.xmax = problemFile["mesh"]["x1max"]
    else:                    self.xmax = config["xmax"]
    if "ymin" not in config: self.ymin = problemFile["mesh"]["x2min"]
    else:                    self.ymin = config["ymin"]
    if "ymax" not in config: self.ymax = problemFile["mesh"]["x2max"]
    else:                    self.ymax = config["ymax"]
    # Convert into floating point numbers
    self.xmin = float(self.xmin)
    self.xmax = float(self.xmax)
    self.ymin = float(self.ymin)
    self.ymax = float(self.ymax)
    # Build array of xy extent to be used in imshow extent
    self.xyextent = np.array([self.xmin,self.xmax,self.ymin,self.ymax])
    # Determine colourmap
    if "cmap" in config: self.cmap = config["cmap"]
    else: self.cmap = "viridis" 
    # Make modifications based on defined unit
    if "xyunits" in config: self.units = config["xyunits"].lower()
    else:                   self.units = "cm"
    # Set image plotting peroperties
    if "contourlevels" in config:
      self.contourLevels = int(config["contourlevels"])
    else:
      self.contourLevels = 64
    # File format extension
    if "extension" in config:
      self.extension = config["extension"]
    else:
      self.extension = "png"
    # Resolution
    if "dpi" in config:
      self.dpi = int(config["dpi"])
    else:
      self.dpi = 150
    # Height and width
    if "plotheight" in config:
      self.plotheight = float(config["plotheight"])
    else:
      self.plotheight = 6
    if "plotwidth" in config:
      self.plotwidth = float(config["plotwidth"])
    else:
      self.plotwidth = 6
    # Determine if indiividual folders are used
    if "writetofolders" in config:
      self.writetofolders = bool(config["writetofolders"])
    else:
      self.writetofolders = False
    # Conversion variables
    cmtom  = 0.01
    cmtoFt = 0.032808399
    cmtoYd = 0.010936133
    cmtoAU = 6.6845871e-14
    cmtoLY = 1.0570008e-18
    cmtoPc = 3.2407793e-19
    if self.units == "cm" or self.units == "cgs":
      self.xyunitlabel = "cm"
    elif self.units == "m" or self.units == "mks" or self.units == "si":
      print("Good choice comrade, CGS units are of course, the devil.")
      self.xyunitlabel = "m"
      self.xyextent *= cmtom
      self.xyscale   = cmtom
    elif self.units == "ft" or self.units == "feet":
      print("You monster.")
      self.xyunitlabel = "Ft"
      self.xyextent *= cmtoFt
      self.xyscale   = cmtoFt
    elif self.units == "yd" or self.units == "yards":
      print("You monster.")
      self.xyunitlabel = "Yd"
      self.xyextent *= cmtoYd
      self.xyscale   = cmtoYd
    elif self.units == "au":
      self.xyunitlabel = "AU"
      self.xyextent *= cmtoAU
      self.xyscale   = cmtoAU
    elif self.units == "ly":
      self.xyunitlabel = "Light Years"
      self.xyextent *= cmtoLY
      self.xyscale   = cmtoLY
    elif self.units == "pc":
      self.xyunitlabel = "Pc"
      self.xyextent *= cmtoPc
      self.xyscale   = cmtoPc
    else:
      print("Unexpected XY unit, defaulting to CGS!")
      self.xyunitlabel = "cm"
      self.xyscale     = 1.0
    # Conversion variables for time axis
    if "showtime" in config:
      if config["showtime"] == True:
        self.tscale = 1.0
        self.tunitlabel = "sec"
        if "tunits" in config:
          self.tunits = config["tunits"].lower()
          if self.tunits == "s":
            self.tscale = 1.0
            self.tunitlabel = "sec"
          elif self.tunits == "min":
            self.tscale = 0.016666667
            self.tunitlabel = "mins"
          elif self.tunits == "hr":
            self.tscale = 0.00027777778
            self.tunitlabel = "hours"
          elif self.tunits == "day":
            self.tscale = 1.1574074e-05
            self.tunitlabel = "days"
          elif self.tunits == "month":
            self.tscale = 3.8026518e-07
            self.tunitlabel = "months"
          elif self.tunits == "yr":
            self.tscale = 3.1688765e-08
            self.tunitlabel = "years"
          else:
            print("Unrecognised label")

    self.xmin *= self.xyscale
    self.xmax *= self.xyscale
    self.ymin *= self.xyscale
    self.ymax *= self.xyscale



def readProblemFile(filename):
  """
  Read in the initial problem file, this is used for a number of things:
    - Finding slices to plot
    - Finding the history file
    - Determining the initial parameters
  Inputs:
    - str filename: filename of problem file
  Outputs:
    - dict problemFile: The parsed problem file, with a 
  """
  problemFile = {}

  with open(filename) as f:
    for line in f:
      try:
        # Separate key from value
        if line.strip()[0] == "<":
          header = line.strip()[1:-1]
          problemFile[header] = {}
        else:
          (key,val) = line.split("=")
          # Strip out comments
          val = val.split("#")[0]
          # Strip extraneous characters from key and val
          key = key.strip()
          val = val.strip()
          # Add line to dictonary
          problemFile[header][key] = val
      except:
        pass
  return problemFile

def calculateAvgMass(problemFile,windno):
  """
  Calculate the average mass of a winds flow
  Inputs:
    - dict problemFile: problem file, containing mass fractions
    - int  windno:      wind number, counting from 1
  Outputs:
    - avgMass: average mass of particles in wind (g)
  """
  protonmass = 1.6726219e-24
  # Find original values from problem file
  xH  = float(problemFile["xH"+str(windno)])
  xHe = float(problemFile["xHe"+str(windno)])
  xC  = float(problemFile["xC"+str(windno)])
  xN  = float(problemFile["xN"+str(windno)])
  xO  = float(problemFile["xO"+str(windno)])
  # Calculate mass contributon
  mH  = xH  * 1.0
  mHe = xHe * 4.0
  mC  = xC  * 12.0
  mN  = xN  * 14.0
  mO  = xO  * 16.0
  # Calculate average mass, in proton masses, convert to grams and return!
  avgMass  = mH + mHe + mC + mN + mO
  avgMass *= protonmass
  return avgMass

def makeFolders(config):
  for plot in config["plots"]:
    if "filename" not in plot:
      folderName = plot["quantity"]
    else:
      folderName = plot["filename"]
    testFolder = glob.glob(folderName)
    if len(testFolder) == 0:
      mkdir(folderName)
  return
  
def calculateTemperature(data,stars):
  """
  Special plot case: calculating temperature, requires multiple input grids, so higher memory usage
  - Uses the ideal gas law equation to find temperature based on density, pressure and
    mass of a particle in the gas
  NOTE:
    - The reason why this code looks so strange is to conserve memory, interpolated hdf5
      datatypes are extremely memory hungry, as such I have written it as to only use two
      arrays at any given time, the final output array and a temporary array used to build
      a specific variable.
    - These temporary arrays are manually deleted as soon as their purpose has been
      fulfilled, I'm not leaving this up to Python's garbage collection.
    - Originally this function used to import everything it needed at initialisation, this
      would cause my computer to immediately crash, I am not taking any chances
    - From what I can tell, there isn't a better way of writing this to be more memory
      efficient, and does not significantly impact the performance of the code.
        - Even if it does, I would prefer memory efficiency to execution time in this
          one case.
  Inputs:
    - str   file:       Filename of datafile being used, fed into GenericImport() 
    - class Commonplot: Common plotting parameters of data
    - list  Stars:      List of stars properties in system, properties stored as class
  Arrays:
    - array colGrid:    Numerical grid containing "colour", ie individual wind contrib
    - array muGrid:     Numerical grid containing gas average mass (g)
    - array rhoGrid:    Numerical grid containing density in CGS units (g cm^-3)
    - array pressgrid:  Numerical grid containing pressure in CGS units (g cm^-1 s^-2)
  Outputs:
    - array tempGrid:   Numerical grid containing gas temperature in CGS units (K)
  """
  kboltz = 1.380658e-16
  # Calculate average mass of particles in cell
  colGrid = data["r0"]
  muGrid  = (stars[0].avgMass * colGrid) + (stars[1].avgMass * (1 - colGrid))
  tempGrid = (data["press"] * muGrid) / (data["rho"] * kboltz)
  return tempGrid

def calculateScalarVelocity(data):
  """
  Function used to calculate speed, this is a special case that requires calculation
  before plotting
  - Uses the equation V = sqrt(vx^2 + vy^2 + vz^2)
  NOTE:
    - The reason why this code looks so strange is to conserve memory, interpolated hdf5
      datatypes are extremely memory hungry, as such I have written it as to only use two
      arrays at any given time, the final output array and a temporary array used to build
      a specific variable.
    - These temporary arrays are manually deleted as soon as their purpose has been
      fulfilled, I'm not leaving this up to Python's garbage collection.
    - Originally this function used to import everything it needed at initialisation, this
      would cause my computer to immediately crash, I am not taking any chances
    - From what I can tell, there isn't a better way of writing this to be more memory
      efficient, and does not significantly impact the performance of the code.
        - Even if it does, I would prefer memory efficiency to execution time in this
          one case.
  Inputs:
    - str   file:       Filename of datafile being used, fed into GenericImport() 
    - class Commonplot: Common plotting parameters of data
  Arrays:
    - array xvelGrid:   Numerical grid containing x axis velocity (cm s^-1)
    - array yvelGrid:   Numerical grid containing y axis velocity (cm s^-1)
    - array zvelGrid:   Numerical grid containing z axis velocity (cm s^-1)
  Outputs:
    - array speedGrid:  Numerical grid scalar velocity (cm s^-1)
  """
  velGrid = (data["vel1"]**2) + (data["vel2"]**2) + (data["vel3"]**2)
  velGrid = np.sqrt(velGrid)
  return velGrid

def calculateKineticEnergy(data):
  """
  Function used to calculate kinetic energy density, this is a special case that requires 
  calculation before plotting
  - Uses the equation E = 1/2 * rho * V^2
  NOTE:
    - The reason why this code looks so strange is to conserve memory, interpolated hdf5
      datatypes are extremely memory hungry, as such I have written it as to only use two
      arrays at any given time, the final output array and a temporary array used to build
      a specific variable.
    - These temporary arrays are manually deleted as soon as their purpose has been
      fulfilled, I'm not leaving this up to Python's garbage collection.
    - Originally this function used to import everything it needed at initialisation, this
      would cause my computer to immediately crash, I am not taking any chances
    - From what I can tell, there isn't a better way of writing this to be more memory
      efficient, and does not significantly impact the performance of the code.
        - Even if it does, I would prefer memory efficiency to execution time in this
          one case.
  Inputs:
    - str   file:       Filename of datafile being used, fed into GenericImport() 
    - class Commonplot: Common plotting parameters of data
  Arrays:
    - array xvelGrid:   Numerical grid containing x axis velocity (cm s^-1)
    - array yvelGrid:   Numerical grid containing y axis velocity (cm s^-1)
    - array zvelGrid:   Numerical grid containing z axis velocity (cm s^-1)
    - array rhoGrid:    Numerical grid containing density (g cm^-3)
  Outputs:
    - array energyGrid: Numerical grid scalar velocity (erg)
  """
  rhoGrid    = data["rho"]
  vel2Grid   = (data["vel1"]**2) + (data["vel2"]**2) + (data["vel3"]**2)
  energyGrid = 0.5 * rhoGrid * vel2Grid
  return energyGrid

def calculateDustDensity(data):
  """
  Function used to calculate kinetic energy density, this is a special case that requires 
  calculation before plotting
  - Uses the equation rhoD = z * rhoG
  NOTE:
    - The reason why this code looks so strange is to conserve memory, interpolated HDF5
      datatypes are extremely memory hungry, as such I have written it as to only use two
      arrays at any given time, the final output array and a temporary array used to build
      a specific variable.
    - These temporary arrays are manually deleted as soon as their purpose has been
      fulfilled, I'm not leaving this up to Python's garbage collection.
    - Originally this function used to import everything it needed at initialisation, this
      would cause my computer to immediately crash, I am not taking any chances
    - From what I can tell, there isn't a better way of writing this to be more memory
      efficient, and does not significantly impact the performance of the code.
        - Even if it does, I would prefer memory efficiency to execution time in this
          one case.
  Inputs:
    - str   file:       Filename of datafile being used, fed into GenericImport() 
    - class Commonplot: Common plotting parameters of data
  Outputs:
    - array rhoDGrid:   Numerical grid containing dust density (g cm^-3)
  """
  rhoDGrid  = data["r1"] * data["rho"]
  return rhoDGrid

def reduceData(data,quant):
  """

  Note: This can be modified to run in 3D, but right I have optimised it for 2d slices
  """
  nindex = np.size(data[quant])
  xdata  = np.empty(nindex)
  ydata  = np.empty(nindex)
  qdata  = np.empty(nindex)
  idx = 0
  for n in range(len(data[quant])):
      for i in range(len(data["x1v"][n])):
        for j in range(len(data["x2v"][n])):
          for k in range(len(data["x3v"][n])):
            xdata[idx] = data["x1v"][n][i]
            ydata[idx] = data["x2v"][n][j]
            qdata[idx] = data[quant][n][k][j][i]
            idx += 1
  return xdata,ydata,qdata

def main(**kwargs):
  # Read yaml config file
  with open(kwargs["config_file"]) as file:
    config = yaml.load(file,Loader=yaml.FullLoader)
  # Read input data
  problemFileName = config["pgen"]
  problemFile = readProblemFile(problemFileName)

  # Generate list of quantities to plot based on data
  quantities = []
  for quant in config["plots"]:
    quantities.append(quant["quantity"])
  
  # Initialise star objects
  WR = Star(problemFile["problem"],1)
  OB = Star(problemFile["problem"],2)
  stars = [WR,OB]

  # Initialise the axes object
  CommonPlot = CommonPlotting(problemFile,config)
  # Generate a list of data files to read
  problemID = problemFile["job"]["problem_id"]
  if kwargs["nfile"] == None:
    # Find all 2d data files in folder
    dataFiles = glob.glob(problemID+".2dxy.*.athdf")
  else:
    # Add specific data file in folder
    n = kwargs["nfile"]
    n = str(n).zfill(5)
    dataFiles = glob.glob(problemID+".2dxy."+n+".athdf")
  if len(dataFiles) == 0:
    print("!!! Program was not able to find any datafiles in folder!")
    print("    Check to see if input string is correct, or if files are present in folder!")
    return 1

  if CommonPlot.writetofolders:
    makeFolders(config)



  # PROCESS DATA
  # Loop through data files
  for file in dataFiles:
    print("Beginning plotting of file = {}".format(file))
    # Import entire dataset, without using interpolation step, triangulation of scattered
    # points is used instead
    data = athena_read.athdf(file,raw = True)
    # Scale x y and z axes, this is done here as there are fewer values to calculate
    # and does not have to be performed recursively
    data["x1v"] *= CommonPlot.xyscale
    data["x2v"] *= CommonPlot.xyscale
    data["x3v"] *= CommonPlot.xyscale

    for plot in config["plots"]:
      # Determine quantity to import
      quant = plot["quantity"]
      # Determine filename
      if "filename" not in plot: plot["filename"] = quant
      exportFilename = "{0}.{1}.{2}".format(file,
                                            plot["filename"],
                                            CommonPlot.extension)
      if CommonPlot.writetofolders:
        exportFilename = "{0}/{1}".format(plot["filename"],exportFilename)
      # Perform specific imports, which require some calculation
      # These are appended to the datastructure
      if quant == "vel":
        data["vel"] = calculateScalarVelocity(data)
      if quant == "ke":
        data["ke"] = calculateKineticEnergy(data)
      if quant == "temp":
        data["temp"] = calculateTemperature(data,stars)
      if quant == "rhod":
        data["rhod"] = calculateDustDensity(data)
      # Determine the title of the plot
      title = r""
      if "title" in config:
        title += r"{}".format(config["title"])
      if "subtitle" in config:
        title += "\n"
        title += r"{}".format(config["subtitle"])
      if "showtime" in config:
        if config["showtime"]:
          time = data["Time"] * CommonPlot.tscale
          tlabel = CommonPlot.tunitlabel
          if "subtitle" not in config:
            title += "\n"
          else:
            title += ", " 
          title = "{0}t = {1:.3f} {2}".format(title,time,tlabel)
      # Import the associated data into three contiguous arrays from meshblocks
      xdata,ydata,zdata = reduceData(data,quant)
      # Scale axes automatically or using predefined min and max
      if "zmin" not in plot: zmin = np.log10(np.min(zdata))
      else: zmin = float(plot["zmin"])
      if "zmax" not in plot: zmax = np.log10(np.max(zdata))
      else: zmax = float(plot["zmax"])

      # Calculating contour levels and colour bar levels
      if "log" in plot:
        if plot["log"] == True:
          log = True
        else:
          log = False
      else:
        log = False
      if log:
        if "e" in str(zmin).lower():
          print(" ! Warning: zmin not in logarithmic form")
          print(" ! Converting zmin from {0} to {1}".format(zmin,np.log10(zmin)))
          zmin = np.log10(zmin)
        if "e" in str(zmax).lower():
          print(" ! Warning: zmax not in logarithmic form")
          print(" ! Converting zmax from {0} to {1}".format(zmax,np.log10(zmax)))
          zmax = np.log10(zmax)
        zmin  = floor(zmin)
        zmax  = ceil(zmax)
        zdel  = (zmax - zmin) + 1
        lcont = np.logspace(zmin,zmax,CommonPlot.contourLevels)
        lcbar = np.logspace(zmin,zmax,zdel)
        zmintri = 10**zmin
        zmaxtri = 10**zmax
        zdata = np.clip(zdata,zmintri,zmaxtri)
      else:
        if "e" not in str(zmin).lower():
          print(" ! Warning: zmin in logarithmic form")
          print(" ! Converting zmin from {0} to {1}".format(zmin,10**zmin))
          zmin = 10**zmin
        if "e" not in str(zmax).lower():
          print(" ! Warning: zmax in logarithmic form")
          print(" ! Converting zmax from {0} to {1}".format(zmax,10**zmax))
          zmax = 10**zmax
        lcont = np.linspace(zmin,zmax,CommonPlot.contourLevels)
        lcbar = np.linspace(zmin,zmax,6)
        zmintri = zmin
        zmaxtri = zmax

      plt.figure(figsize=(CommonPlot.plotwidth,CommonPlot.plotheight))
      ax = plt.gca()
      ax.set_aspect(1)
      # PERFORM PLOTTING
      if log:
        plt.tricontourf(xdata,ydata,zdata,
                        levels=lcont,
                        norm=LogNorm(vmin=zmintri,vmax=zmaxtri),
                        cmap=CommonPlot.cmap)
      else:
        plt.tricontourf(xdata,ydata,zdata,
                        levels=lcont,
                        cmap=CommonPlot.cmap)


      plt.colorbar(label=r"{}".format(plot["label"]),
                   ticks=lcbar)
      plt.xlabel("X ({})".format(CommonPlot.xyunitlabel))
      plt.ylabel("Y ({})".format(CommonPlot.xyunitlabel))
      plt.xlim([CommonPlot.xmin,CommonPlot.xmax])
      plt.ylim([CommonPlot.ymin,CommonPlot.ymax])
      plt.title(title)
      plt.savefig(exportFilename,dpi=CommonPlot.dpi)
      plt.clf()
      print(" + Finished {}".format(quant))
    print(" ! Finished all plots of {}!".format(file))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Process 2D and hst data from a folder of Athena outputs.\nPositional arguments and flags can  be used, if all arguments are left blank, a config file will instead be used.")
  parser.add_argument("config_file",
                       help="name of yaml config file to be used.")
  parser.add_argument("-n",
                      "--nfile",
                      type=int,
                      default=None,
                      help="ID number of data file to process, if left blank, all files in folder will be processed, using this allows for easier parallelisation on SGE submission scripts.")
  args = parser.parse_args()
  main(**vars(args))
