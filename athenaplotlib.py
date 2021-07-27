from numba import njit,prange
import numpy as np
from math import pi,sin,cos,sqrt,atan2,acos
import yaml 

# Import matplotlib plotting libraries
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# Import astropy libraries
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
# Import Athena++ HDF5 library
import athena_read

class Star:
  def __init__(self,problem_file,star_no):
    # Determine if star is defined numerically or with string
    if type(star_no) == str:
      if star_no.upper() == "WR":
        star_no = 1
      elif star_no.upper() == "OB":
        star_no = 2
    # Attempt to read in stellar parameters from problem file
    try:
      # Import stellar properties
      self.mdot = readParameter(problem_file["problem"],
                                "mdot"+str(star_no),
                                quit_on_fail=True)
      self.vinf = readParameter(problem_file["problem"],
                                "vinf"+str(star_no),
                                quit_on_fail=True)
      self.mass = readParameter(problem_file["problem"],
                                "vinf"+str(star_no),
                                quit_on_fail=True)
      # Import wind abundances
      elements = ["H","He","C","N","O"] # Element names
      default_abundances = [0.705,0.275,0.003,0.001,0.010] # In a pinch, use solar abundances
      abundances = [0.,0.,0.,0.,0.]
      for n in range(len(elements)):
        abundances[n] = readParameter(problem_file["problem"],
                                      "x"+elements[n]+str(star_no),
                                      fallback=default_abundances[n])
        abundances[n] = float(abundances[n]) # Ensure saved as float
      # Import orbital properties
      self.period   = readParameter(problem_file["problem"],
                                    "period",
                                    quit_on_fail=True)
      self.ecc      = readParameter(problem_file["problem"],
                                    "ecc",
                                    quit_on_fail=True)
      self.phaseoff = readParameter(problem_file["problem"],
                                    "phaseoff",
                                    quit_on_fail=True)
    except:
      print("! Could not read in star parameters from problem file!")
      raise
    # Perform unit conversion on basic parameters
    self.mdot_mdotyr = float(self.mdot)
    self.vinf     = float(self.vinf)
    self.mass     = float(self.mass)
    self.period   = float(self.period)
    self.ecc      = float(self.ecc)
    self.phaseoff = float(self.phaseoff)
    self.mdot     = self.mdot_mdotyr * Conv.msolyr_to_gsec
    self.avgm     = calcAvgMass(abundances)
    self.mu       = self.avgm / Const.m_P
    # Add a blank list for position
    self.pos = []
    return


class Const:
  m_P = 1.6726219e-24
  m_e = 9.1093837e-28
  kB  = 1.3806490e-16
  G   = 6.6743000e-12
  # Constants that are arrays
  solar_abundances = [0.705,0.275,0.003,0.001,0.010]

class Conv:
  """
  Common conversion factors used in Athena
  """
  # Common conversions to CGS
  yr_to_sec      = 31556926
  msol_to_g      = 1.9884099e+33
  msolyr_to_gsec = 6.3010252e+25
  # Common conversions from CGS
  # CGS -> SI
  cm_to_m  = 0.01
  ba_to_pa = 0.1
  erg_to_j = 1e-07
  # CGS -> Standard
  cm_to_au = 6.6845871e-14
  cm_to_ly = 1.0570008e-18
  cm_to_pc = 3.2407793e-19
  # CGS -> Nonstandard
  cm_to_smoot = 0.0058761312

def calcAvgMass(abundances):
  """
  Calculate average mass of star wind, using abundances
  """
  m_P = Const.m_P
  elemental_masses = [1.0,4.0,12.0,14.0,16.0]
  avg_mass = 0.
  for n in range(len(elemental_masses)):
    m_el = elemental_masses[n]
    x_el = abundances[n]
    m_c  = m_el * x_el   # Calculate mass contribution of element
    avg_mass += m_c      # Append mass to total average mass
  avg_mass *= m_P # Convert from proton masses to grams
  return avg_mass

def calcDsep(star1,star2):
  dsepx = star1.pos[0] - star2.pos[0]
  dsepy = star1.pos[1] - star2.pos[1]
  dsepz = star1.pos[2] - star2.pos[2]
  dsep  = (dsepx**2 + dsepy**2 + dsepz**2)**0.5
  return dsep

def calcEta(star1,star2):
  """
  Calculate wind momentum ratio between binary pair

  This assumes that star1 has greateset wind momentum, and is dominant
  In the case of an WR+OB binary pair:
  eta = (mdot_OB * vinf_OB) / (mdot_WR * vinf_WR)
  """
  eta = (star2.mdot * star2.vinf) / (star1.mdot * star1.vinf)
  return eta

def calcWCRApexPosition(star1,star2):
  """
  Calculate position of WCR apex, simple geometric argument, assuming cartesian coordinates
  Inputs:
    Star star1: First star object, typically WR 
    Star star2: Second star object, typically OB
  Outputs:
    list pos:   x,y,z indexed position of WCR apex
  """
  # First, calculate the distance between each star
  dsep = calcDsep(star1,star2)
  # Calculate Momentum ratio 
  eta  = calcEta(star1,star2)
  # Calculate distance from WR to WCR apex, Usov 1991
  r_wr = (1/(1+eta**0.5)) * dsep
  # Convert distances to a fraction
  frac = dsep/r_wr
  # Calculate angle between WCR and OB on
  pos  = []
  pos.append(star1.pos[0] + (star2.pos[0] - star1.pos[0])*frac)
  pos.append(star1.pos[1] + (star2.pos[1] - star1.pos[1])*frac)
  pos.append(star1.pos[2] + (star2.pos[2] - star1.pos[2])*frac)
  return pos

def calcPhase(star,time):
  """
  Calculate the phase of an orbit, very simple calculation but used quite a lot
  """
  period = star.period
  phase  = time/period
  return phase

def calcOrbit(star1,star2,time):
  """
  Calculate the orbit of a stellar 
  This assumes that stellar orbit is always aligned parallel to z axis at z=0, hence only x,y positions are used
  I am not proud of this one, as it is sloppily ported over from a previous script, which in turn was sloppily ported over from a C program written 10 years ago, no I am not cleaning it up.
  """
  G        = Const.G
  period   = star1.period
  ecc      = star1.ecc
  phaseoff = star1.phaseoff
  m1       = star1.mass
  m2       = star2.mass
  msol     = Conv.msol_to_g
  phase    = time/period
  phase   += phaseoff        # Append phase offset phase
  # Sloppily paste in code here
  phi = 2.0*pi*phase
  E = phi
  dE = (phi - sin(E) + ecc)/(1.0-ecc * cos(E))
  while (abs(dE) > 1.0e-10):
      E = E + dE
      dE = (phi - E + ecc*sin(E))/(1.0-ecc*cos(E))
  sii = sqrt(1.0 - ecc*ecc)*sin(E)/(1.0 - ecc*cos(E))
  coi = (cos(E)-ecc)/(1.0-ecc*cos(E))
  theta = atan2(sii,coi)
  if (theta < 0.0): theta = 2.0*pi + theta
  rrel = (1.0 - ecc*cos(E))
  M = (m2**3)/((m1+m2)**2)
  a1 = ((G*M*period*period)/(4.0*pi*pi))**(1/3) * (msol)**(1/3)
  xs1 = a1*rrel*cos(theta)
  ys1 = a1*rrel*sin(theta)
  xs1 = -xs1
  ys1 = -ys1
  M = (pow(m1,3)/pow(m1+m2,2))*msol
  a2 = pow(G*M*period*period/(4.0*pi*pi),1.0/3.0)
  xs2 = a2*rrel*cos(theta)
  ys2 = a2*rrel*sin(theta)
  xs2 = xs2
  ys2 = ys2
  star1pos = [xs1,ys1,0.0]
  star2pos = [xs2,ys2,0.0]
  return star1pos,star2pos

def readParameter(dictionary,key,fallback=None,quit_on_fail=False):
  """
  Define a variable if it is contained in the dictionary, if not use a fallback or quit outright.

  Inputs:
    - dict dictionary:   dictionary to read
    - str  key:          key to read
    - val  fallbac:      fallback to return instead, default Nonetype
    - bool quit_on_fail: if True, halt program if key not present,
                         default False
  """
  try:
    parameter = dictionary[key]
  except:
    if quit_on_fail == True:
      print("! Essential variable not found in dictionary, exiting.")
      raise
    else:
      parameter = fallback
      pass
  return parameter


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


def reduce(grid,quants,avgmass_WR = None,avgmass_OB = None):
  # Sub-Functions used by reduction function
  @njit(parallel=True)
  def reduceXYZ(xgrid,ygrid,zgrid,nn,ni,nj,nk,nr):
    xyz_reduced = np.zeros((3,nr))
    for n in prange(nn):
      for i in range(ni):
        for j in range(nj):
          for k in range(nk):
            idx = n + (i*nn) + (j*nn*ni) + (k*nn*ni*nj)
            xyz_reduced[0,idx] = xgrid[n][i]
            xyz_reduced[1,idx] = ygrid[n][j]
            xyz_reduced[2,idx] = zgrid[n][k]
    return xyz_reduced
  @njit(parallel=True)
  def reduceQ(qgrid,nn,ni,nj,nk,nr):
    q_reduced = np.zeros(nr)
    for n in prange(nn):
      for i in range(ni):
        for j in range(nj):
          for k in range(nk):
            idx = n + (i*nn) + (j*nn*ni) + (k*nn*ni*nj)
            q_reduced[idx] = qgrid[n][k][j][i]
    return q_reduced
  # Execution
  # Check to see if quantity is in the form of a list, if not, make it a list
  if type(quants) == str:
    quants = [quants]
  # Slice positional arrays
  xgrid = grid["x1v"]
  ygrid = grid["x2v"]
  zgrid = grid["x3v"]
  # Determine length of positional arrays
  nn = len(xgrid)    # Total number of meshblocks
  ni = len(xgrid[0]) # Number of cells in x direction in meshblock
  nj = len(ygrid[0]) # Number of cells in y direction in meshblock
  nk = len(zgrid[0]) # Number of cells in z direction in meshblock
  # Determine required length of reduced grid
  nr = nn*ni*nj*nk
  # Initialise reduced grid dictionary
  reduced_array = dict()
  # Reduce dimensional arrays
  reduced_array["xyz"]  = reduceXYZ(xgrid,ygrid,zgrid,nn,ni,nj,nk,nr)
  # Reduce quantity arrays
  # Determine which quantities are "canonical" (included in HDF5 file) or need to be derived
  # See https://github.com/PrincetonUniversity/athena-public-version/wiki/HDF5-Format#names-of-quantities

  c_quants = ["rho","press","vel1","vel2","vel3","r0","r1","r2"]
  derived_quants = []
  for quant in quants:
    if quant in c_quants:
      reduced_array[quant] = reduceQ(grid[quant],nn,ni,nj,nk,nr)
    else:
      derived_quants.append(quant)

  # Calculate quantity, these need to be added manually, as they come up
  # Currently supported:
  # - Temperature (temp)
  # - Dust density (rhod)
  for quant in derived_quants:
    if quant == "temp":
      if avgmass_WR == None or avgmass_OB == None:
        print("Accurate temperature calculation requires star average masses which have not been provided when calling reduce_data(), defaulting to solar abundance average mass for both winds")
        if avgmass_WR == None: avgmass_WR = calcAvgMass(Const.solar_abundances)
        if avgmass_OB == None: avgmass_OB = calcAvgMass(Const.solar_abundances)
      if "press" not in quants:
        reduced_array["press"] = reduceQ(grid["press"],nn,ni,nj,nk,nr)
      reduced_array[quant] = calcTemp(reduced_array,avgmass_WR,avgmass_OB)
    if quant == "rhod":
      reduced_array[quant] = calcRhoD(reduced_array)
  
  # Finish up and return reduced array!    
  return reduced_array

@njit(parallel=True)
def calcMagnitudeArray(xarr,yarr,zarr):
  ni = len(xarr)
  magarr = np.zeros(ni)
  for i in prange(ni):
    x2 = xarr[i] * xarr[i]
    y2 = yarr[i] * yarr[i] 
    z2 = zarr[i] * zarr[i]
    m2 = x2 + y2 + z2
    m  = m2 ** 0.5
    magarr[i] = m
  return magarr

def calcRhoD(reduced_array):
  @njit(parallel=True)
  def performCalc():
    rhod_arr = np.zeros(ni)
    for i in prange(ni):
      # Calculate dust density
      rho  = rho_arr[i]
      z    = z_arr[i]
      rhod = rho * z
      # Add to array
      rhod_arr[i] = rhod
    return rhod_arr
  ni       = len(reduced_array["rho"])
  rho_arr  = reduced_array["rho"]
  z_arr    = reduced_array["r1"]
  rhod_arr = performCalc()
  return rhod_arr

def calcTemp(reduced_array,WRavgmass,OBavgmass):
  @njit(parallel=True)
  def performCalc():
    temp_arr = np.zeros(ni)
    for i in prange(ni):
      col   = col_arr[i]
      mu    = (WRavgmass * col) + (OBavgmass * (1-col))
      rho   = rho_arr[i]
      press = press_arr[i]
      temp  = (press * mu) / (rho * kB)
      # Write to array
      temp_arr[i] = temp
    return temp_arr
  # Get constants
  kB = Const.kB
  ni        = len(reduced_array["rho"])
  rho_arr   = reduced_array["rho"]
  press_arr = reduced_array["press"]
  col_arr   = reduced_array["r0"]
  temp_arr  = performCalc()
  return temp_arr

def calcDistFromPoint(reduced_array,point_pos):
  @njit(parallel=True)
  def performCalc(xyz_data,xpos,ypos,zpos):
    """
    Calculate distances from star, ctypes+parallel for speed
    """
    rarr = np.zeros(ni)
    for i in prange(ni):
      x2 = (xyz_data[0][i] - xpos) ** 2
      y2 = (xyz_data[1][i] - ypos) ** 2
      z2 = (xyz_data[2][i] - zpos) ** 2
      r2 = x2 + y2 + z2
      r  = r2 ** 0.5
      rarr[i] = r
    return rarr
  ni = len(reduced_array["xyz"][0])
  xpos = point_pos[0]
  ypos = point_pos[1]
  zpos = point_pos[2]
  xyz_data = reduced_array["xyz"]
  r_array  = performCalc(xyz_data,xpos,ypos,zpos)
  return r_array

def detIfInWCR(reduced_array,star_WR,star_OB,scale=1.0):
  @njit(parallel=True)
  def calcSingleWindDensity(rarr,mdot,vinf,ni):
    """
    """
    swrhoarr = np.zeros(ni)
    for i in prange(ni):
      swrhoarr[i] = mdot/(4.0*pi*rarr[i]*rarr[i]*vinf)
    return swrhoarr

  @njit(parallel=True)
  def WCRdetSingleWind(rhoarr,colarr,swrhoarrWR,swrhoarrOB,ni,scale):
    in_wcr_array = np.zeros(ni)
    mm = 0.
    for i in prange(ni):
      if colarr[i] > mm:
        if rhoarr[i] > scale * swrhoarrWR[i]:
          incwb = True
        else:
          incwb = False
      elif colarr[i] < mm:
        if rhoarr[i] > scale * swrhoarrOB[i]:
          incwb = True
        else:
          incwb = False
      else:
        incwb = False
      in_wcr_array[i] = incwb
    return in_wcr_array 

  ni = len(reduced_array["rho"])
  # Use overdensity method to calculate 
  try:
    # Grab some constants from non numpy objects
    vinfWR = star_WR.vinf
    mdotWR = star_WR.mdot
    vinfOB = star_OB.vinf
    mdotOB = star_OB.mdot
    # Process data using functions
    sw_array_WR = calcSingleWindDensity(reduced_array["r"],mdotWR,vinfWR,ni)
    sw_array_OB = calcSingleWindDensity(reduced_array["r"],mdotOB,vinfOB,ni)
    # Finally, wrap up and calculate truth array
    in_wcr_array = WCRdetSingleWind(reduced_array["rho"],
                                    reduced_array["r0"],
                                    sw_array_WR,sw_array_OB,
                                    ni,scale)
  except:
    print("Could not determine if values were in WCR, most likely missing keys \"d2\" and \"rho\" from reduced_array!")
    raise 
  return in_wcr_array

@njit(parallel=True)
def scaleArray(array,scale):
  ni = len(array)
  scaled_array = np.zeros(ni)
  for i in prange(ni):
    scaled_array[i] = array[i] * scale
  return scaled_array

def showCWB(ax,slice_data,xyscale,alpha=1.0):
  xdata = scaleArray(slice_data["xyz"][0],xyscale)
  ydata = scaleArray(slice_data["xyz"][1],xyscale)
  zdata = slice_data["incwb"]
  ax.tricontour(xdata,ydata,zdata,levels=1,colors="red",linewidths=0.5,alpha=alpha)
  return ax

def buildAxisLabel(label,unit):
  if unit == "" or unit == None:
    axis_label = "{0}".format(label)
  else:
    axis_label = "{0} ({1})".format(label,unit)
  return axis_label

from matplotlib.cm import ScalarMappable

def plotSlice(fig,ax,xdata,ydata,zdata,plot_config,global_config,alpha=1.0):
  quant   = readParameter(plot_config,"quantity",quit_on_fail=True)
  islog   = bool(readParameter(plot_config,"log",fallback=False))
  xyscale = float(readParameter(global_config,"xyscale",fallback=1.0))
  zscale  = float(readParameter(plot_config,"zscale",fallback=1.0))
  clevs   = int(readParameter(global_config,"contourlevels",fallback=256))
  cmap    = readParameter(global_config,"colourmap",fallback="viridis")
  # Build colourbar label
  clabel = readParameter(plot_config,"label",fallback="")
  cunit  = readParameter(plot_config,"unit")
  if cunit != None:
    clabel = "{0} ({1})".format(clabel,cunit)
  # Build x and y axes labels
  xyunit = readParameter(global_config,"xyunit")
  xlabel = "X"
  ylabel = "Y"
  if xyunit != None:
    xlabel = "{0} ({1})".format(xlabel,xyunit)
    ylabel = "{0} ({1})".format(ylabel,xyunit)

  # Scale xy axes for position
  if xyscale != 1.0:
    xdata = scaleArray(xdata,xyscale)
    ydata = scaleArray(ydata,xyscale)
  # Scale quantity axes for colour map
  if zscale != 1.0:
    zdata = scaleArray(zdata,zscale)

  # Calculate minimum and maximum values for plotting
  zmin = float(readParameter(plot_config,"zmin",fallback=0.))
  zmax = float(readParameter(plot_config,"zmax",fallback=0.))
  # # Check to see if zmin/zmax are not stated, if so, calculate minimum and maximum, this is faster than defining it as a fallback in readparameter()
  # if zmin == 0.: zmin = np.log10(np.min(data[quant]))
  # if zmax == 0.: zmax = np.log10(np.max(data[quant]))

  if islog:
    lcont = np.logspace(zmin,zmax,clevs)
    zdel = int((zmax - zmin) + 1)
    lcbar = np.logspace(zmin,zmax,zdel)
    ax.tricontourf(xdata,ydata,zdata,
                   levels=lcont,
                   norm=LogNorm(vmin=10**zmin,vmax=10**zmax),
                   cmap=cmap)
    fig.colorbar(ScalarMappable(norm=LogNorm(vmin=10**zmin,vmax=10**zmax), cmap=cmap), ax=ax,ticks=lcbar,label=clabel)
  else:
    zmin = 10**zmin
    zmax = 10**zmax
    lcont = np.linspace(zmin,zmax,clevs)
    lcbar = np.linspace(zmin,zmax,6)
    ax.tricontourf(xdata,ydata,zdata,
                   levels=lcont,
                   cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(ScalarMappable(norm=LogNorm(vmin=zmin,vmax=zmax), cmap=cmap), ax=ax,ticks=lcbar,label=clabel)
  
  return fig,ax

def plotQuantVsDistScatter(ax,rdata,qdata,plot_config,global_config):
  clabel = readParameter(plot_config,"label",fallback="")
  cunit  = readParameter(plot_config,"unit")
  dlabel = "Distance"
  dunit  = readParameter(global_config,"xyunit",fallback="")
  dpi    = int(readParameter(global_config,"dpi",150))
  cmap   = readParameter(global_config,"colourmap",fallback="viridis")
  # Scale data
  rscale = float(readParameter(global_config,"xyscale",fallback=1.0))
  rdata  = scaleArray(rdata,rscale)
  qscale = float(readParameter(plot_config,"yscale",fallback=1.0))
  qdata  = scaleArray(qdata,qscale)
  norm   = ImageNormalize(vmin=1,vmax=100,stretch=LogStretch())

  ax.scatter_density(rdata,
                     qdata,norm=norm,
                     rasterized=True,dpi=150,cmap=cmap)

  # ax.scatter(rdata,qdata,marker=".",alpha=0.1)
  ax.set_xlabel(buildAxisLabel(dlabel,dunit))
  ax.set_ylabel(buildAxisLabel(clabel,cunit))
  ax.set_yscale("log")
  return ax

def readDataFile(filename):
  """
  Read in an HDF5 datafile, this is just a wrapper
  """
  data = athena_read.athdf(filename,raw = True)
  return data

def readConfigFile(filename):
  """
  Read in YAML config file, this is also a wrapper
  Inputs:
    - str filename: Filename of YAML file
  Outputs:
    - dict config:  Dictionary containing data
  """
  try:
    # Load entire config file into memory
    with open(filename) as config_file:
      config = yaml.load(config_file,Loader=yaml.FullLoader)
  except:
    print("Could not read in config file!")
    raise
  return config