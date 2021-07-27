
import athenaplotlib as apl

from athenaplotlib import Conv,Const


import argparse
from numba import njit,prange
import numpy as np
from scipy.stats import gaussian_kde

import mpl_scatter_density

import matplotlib.gridspec as gridspec

# Make the norm object to define the image stretch
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

@njit
def linearPlotDataInWCR(r_array,quant_array,inWCR_array):
  """
  Build linear plot data for bottom graph, 
  Outputs:
    WCR_array: Plottable array containing x data, r, and y data, the desired quantity
  """
  ncellsinwcr = np.count_nonzero(inWCR_array)
  plot_array  = np.zeros((2,ncellsinwcr))
  idx = 0
  for i in range(len(inWCR_array)):
    if inWCR_array[i] == True:
      plot_array[0,idx] = r_array[i]
      plot_array[1,idx] = quant_array[i]
      idx += 1
  return plot_array

@njit
def slicePlotDataInWCR(x_array,y_array,z_array,inWCR_array):
  """
  Ways to improve on this:
  include a translation array containing index number of element, allows for parallel transfer of memory and minimises reuse, but this is fine for now
  """
  ninWCR = np.count_nonzero(inWCR_array)
  slice_array = np.zeros((3,ninWCR))
  idx = 0
  for i in range(len(inWCR_array)):
    if inWCR_array[i] == True:
      slice_array[0,idx] = x_array[i]
      slice_array[1,idx] = y_array[i]
      slice_array[2,idx] = z_array[i]
      idx += 1
  return slice_array

def plotQuantity(grid_image_data,wcr_image_data,graph_data,config,quant):
  """
  Plot a quantity for this script, script

  inputs:
    array grid_image_data: data for upper plot, 2d slice with a quantity
    array wcr_image_data:  WCR data for upper plot, overlaid on top
    array graph_data:      distance/quantity plot data for lower graph
    dict  config:          config file dictionary
    str   quant:           quantity name, must be in all arrays
  """
  # Enable LaTeX plotting backend, requires TeXLive to be installed
  plt.rcParams.update({
      "text.usetex": True,
      "font.family": "serif",
  })

  fig,axs = plt.subplots(2)


  return 

def processQuantity():
  # 
  return

def processPlot(config,problem,data,base_filename):
  # Get simulation time
  time    = float(apl.readParameter(data,"Time",fallback=0.0))
  # Import star properties and calculate orbital position of stars
  star_WR = apl.Star(problem,"WR")
  star_OB = apl.Star(problem,"OB")
  star_WR.pos,star_OB.pos = apl.calcOrbit(star_WR,star_OB,time)

  # Determine where centerpoint co-ordinate for distance calculations is
  try:
    centerpoint = config["centerpoint"].lower()
    if   centerpoint == "apex":
      center_pos = apl.calcWCRApexPosition(star_WR,star_OB)
    elif centerpoint == "barycenter":
      center_pos = [0.0,0.0,0.0]
    elif centerpoint == "wr":
      center_pos = star_WR.pos
    elif centerpoint == "ob":
      center_pos = star_OB.pos
    else:
      print("Invalid centerpoint argument, defaulting to 0,0,0")
      center_pos = [0.0,0.0,0.0]
  except:
    print("Missing centerpoint argument in config file")
    raise

  # Reduce data from data file
  # Build list of quantities needed
  quantities = ["rho","vel1","vel2","vel3","r0"] # These values always needed
  for plot in config["plots"]:
    quantities.append(plot["quantity"])
  # Deduplicate
  quantities = list(dict.fromkeys(quantities))
  # Finally, reduce data
  reduced_data = apl.reduce(data,quantities,
                            avgmass_WR = star_WR.avgm,
                            avgmass_OB = star_OB.avgm)
  # Calculate distance from centerpoint for each cell
  wcr_method = apl.readParameter(config,"wcr_method",fallback="density")
  wcr_scale  = float(apl.readParameter(config,"wcr_scale",fallback=1.0))
  reduced_data["r"] = apl.calcDistFromPoint(reduced_data,center_pos)

  reduced_data["incwb"] = apl.detIfInWCR(reduced_data,
                                         star_WR, star_OB,
                                         scale=wcr_scale)

  


  xyscale = float(apl.readParameter(config,"xyscale",fallback=1.0))

  for plot in config["plots"]:
    # Build slice subset of data contained in 
    wcr_linear = linearPlotDataInWCR(reduced_data["r"],
                                     reduced_data[plot["quantity"]],
                                     reduced_data["incwb"])
    wcr_slice  = slicePlotDataInWCR(reduced_data["xyz"][0],
                                    reduced_data["xyz"][1],
                                    reduced_data[plot["quantity"]],
                                    reduced_data["incwb"])
    # Build plotting environment
    cmap = apl.readParameter(config,"colourmap",fallback="viridis")
    # Use LaTeX environment
    plt.rcParams.update({
      "text.usetex": True,
      "font.family": "serif",
    })
    # Two figures being plotted
    fig = plt.figure(figsize=(5,8))
    gs  = gridspec.GridSpec(2, 1,height_ratios=[2,1])
    # gs = gridspec.GridSpec(1,1)
    slice_plot   = fig.add_subplot(gs[0])
    slice_plot.set_aspect("equal")
    apl.plotSlice(fig,slice_plot,
                  reduced_data["xyz"][0],
                  reduced_data["xyz"][1],
                  reduced_data[plot["quantity"]],
                  plot,
                  config)
    apl.showCWB(slice_plot,reduced_data,xyscale,alpha=0.5)

    vmin = apl.readParameter(plot,"zmin")
    vmax = apl.readParameter(plot,"vmax")
     # Should be able to move this to plotslice() with fig!

    scatter_plot = fig.add_subplot(gs[1],projection="scatter_density")
    apl.plotQuantVsDistScatter(scatter_plot,
                               wcr_linear[0],
                               wcr_linear[1],
                               plot,config)
    suptitle = apl.readParameter(config,"suptitle",fallback="")
    fig.suptitle(suptitle)
    # Finish up and save figure
    quant_name = apl.readParameter(plot, "filename", quit_on_fail=True)
    extension = apl.readParameter(config, "extension",fallback="png")
    dpi = int(apl.readParameter(config, "dpi",fallback=150))
    filename = "{0}-{1}.{2}".format(base_filename,quant_name,extension)
    plt.savefig(filename,dpi=300)


  # plot_array = linearPlotDataInWCR(data_reduced["d2"],
  #                                  data_reduced[quant],
                                  #  data_reduced["inWCR"])
  
  # r2arr = apl.calculateDistFromStar(data_red)
  # smooth_wind_density = apl.calculateSingleWindDensity(r2arr,mdot1,vinf1)
  # true_index = apl.detInWCR(data_red[3],smooth_wind_density)

  # plotarr = apl.buildPlotArray(data_red[3],r2arr,true_index)


  # norm = ImageNormalize(vmin=1., vmax=1000., stretch=LogStretch())
  # fig = plt.figure()
  # ax = fig.add_subplot(1,1,1,projection="scatter_density")
  # density = ax.scatter_density(plot_array[0],plot_array[1],norm=norm)
  # fig.colorbar(density)
  # ax.set_yscale("log")
  # plt.show()
    
    # Build plot array for the 
    


  # plt.hist2d(plotarr[0],plotarr[1],(100,100),cmap=plt.cm.jet,norm=LogNorm())
  # plt.scatter(plotarr[0],plotarr[1],alpha=0.1)
  # plt.yscale("log")
  # plt.show()



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Process 2D and hst data from a folder of Athena outputs.\nRequires a config file and a positional argument of at least one config file, additional prefixes and directories can also be added")
  parser.add_argument("config_filename",
                      help="Name of config file used for this program")
  parser.add_argument("athena_input_filename",
                      help="Name of Athena config file, needed for config")
  parser.add_argument("HDF5_filename",
                      help="Name of HDF5 grid file, extension .athdf")
  parser.add_argument("output_filename",
                      help="Filename of output, will be stored in the form <output_filename>-<quantity>.<extension>")
  args,overrides = parser.parse_known_args()
  config  = apl.readConfigFile(args.config_filename)
  problem = apl.readProblemFile(args.athena_input_filename)
  data    = apl.readDataFile(args.HDF5_filename)
  output_filename = args.output_filename
  # Override configs
  if len(overrides) > 0:
    print("! Overriding config file with program arguments")
    for arg in overrides:
      try:
        key,val = arg.split("=",1)
        config[key] = val
        print("> {0} = \"{1}\"".format(key,val))
      except:
        pass
    print("! Done!")
  processPlot(config,problem,data,output_filename)