#!/usr/bin/env python3
import numpy as np
import athena_plot
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

class Simulation:
  """
  Class containing simulation parameters and data. Main building block of this programme.

  Inputs:
    str    filename:   Filename to read, typically included as an argument when running
                       this programme
  Variables:
    str    problem_ID: Problem ID, common filename for all data spawned in this 
                       particular simulation
    str    label:      Plotting label, used in the legend of all plots
  Arrays:
    dict   pgen:       Contains entire pgen file in the form of a dictionary,
                       divided up in the form [block][parameter]
    pandas data:       Contains numerical data from the history file, used for plotting
                       columns are labelled, data is automatically repaired by removing
                       data from "dead" simulations, which were closed before a checkpoint
                       or crashed at some point and were restarted
  """
  def __init__(self,filename):

    # Atrocious text parser, finding the root directory for each filename
    if os.path.exists(filename):
      abspath = os.path.abspath(filename)
      self.dir = ""
      for n in abspath.split("/")[:-1]:
        if len(n) > 0:
          self.dir += ("/"+n)
    else:
      print("Shit busted")
      sys.exit()

    self.absfilename  = self.dir + "/" + filename
    self.pgen         = readProblemFile(self.absfilename)
    self.problem_ID   = self.pgen["job"]["problem_id"]
    history_file_name = self.problem_ID+".hst"
    self.absprobname  = self.dir + "/" + history_file_name
    self.data         = readData(self.absprobname)
    # Assing a label to the simulation, can be manually entered in the <comment> block
    # of the pgen file, if not it will use the problem_ID cariable in the <job> block
    if "label" in self.pgen["comment"]:
      self.label = self.pgen["comment"]["label"]
    else:
      self.label = self.problem_ID
    return

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
          (key,val) = line.split("=",1)
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

def readHeader(filename,header_location):
  """
  Inputs:
  - str  filename: Filename of history file
  Outputs:
  - list header:   list of variables used in dataset, this is used for retrieving indexes 
    when plotting
  """
  file = open(filename)
  for i, line in enumerate(file):
    if i == header_location:
      raw_header = line.strip()[1:]
      header = raw_header.split(",")
      for n in range(len(header)):
        header[n] = header[n].strip()
      break
  return header 

def maskData(data):
  """
  - Occasionally, the simulation may crash due to a Riemann solver failure
  - This is typically fixed by a restart from a checkpoint with a lower courant number
  - However, the data is still there in the history file
  - This function masks out incorrect data, by finding areas where "time travel" (time
    value going back on itself) has occurred
  - Algorithm could be more optimal/use pandas datatypes, but is fine for most cases
  - Looped until data is monotonic, so can handle multiple crash-restart events
  """
  while data["time"].is_monotonic_increasing == False:
    # Run through data to find section where data is not behaving monotonically
    cum_max         = 0 # Cumulative maximum time
    i               = 0 # Index counter
    tl_finish       = 0 # Time when normal timeline interrupted 
    tl_finish_index = 0 # Index when normal timeline interrupted
    for t in data["time"]:
      if t < cum_max:
        if tl_finish_index == 0:
          tl_finish_index = i
          tl_finish       = t
      else:
        if tl_finish_index != 0:
          break
        cum_max = t
      i += 1
    # Run through data again to find start of first time loop
    i = 0
    for t in data["time"]:
      if t > tl_finish:
        tl_start_index = i
        break
      i += 1
    # Delete first  recurrence, so that data works correctly
    data = data.drop(data.index[tl_start_index:tl_finish_index])
  return data

def skipToLastEntry(filename):
  """
  - One quirk of Athena++ history files is that if a new simulation is run with the same
    history filename, data is appended to the end, rather than overwriting the file. 
  - Because I frequently forget to remove all the data from a directory, this data 
    remains, rather than fixing this, I'll just write a function to remove the data from 
    plotting.
  - This function finds all instances where a new athena file has been appended, and
    redurns the row indexes of the header files and start of data for that instance.
  - This is performed by finding the initialisation string of the history file by going
    through the entire dataset, and updating the location indices when a new instance is
    found.
  Potential modification, include in config file a way of changing which data entry to use
  Though right now I cannot think of a reason to do this

  Inputs:
  - str filename: Filename of history file
  Outputs:
  - int header_location: Location of the header row for most recent athena iteration
  - int data_location:   Location of the first data row for most recent athena iteration
  """
  fp = open(filename)
  for i, line in enumerate(fp):
    if line.strip() == "# Athena++ history data":
      header_location = i+1
      data_location   = i+2
  return header_location,data_location

def readData(filename):
  """
  Inputs:
  - str filename: Filename of history file
  Outputs:
  - df data:   Entire dataset from 
  """
  # First, find number of rows to skip
  header_location,data_location = skipToLastEntry(filename)
  # Read in header, then data, then merge into a single pandas array
  header       = readHeader(filename,header_location)
  data         = pd.read_csv(filename,comment="#",skiprows=data_location)
  data.columns = header
  # Check to see if data contains any restarts, mask out data between restarts
  if data["time"].is_monotonic_increasing == False:
    data = maskData(data)
  return data

def evalQuantity(plot,sim):
  """
  Evaluate the quantity of plot data
  This plotting program allows for arbitrary mathematical operations in the config
  file, these are 
  Multiple equations are allowed for each plot, allowing for easy comparative
  operations, each evaluation must be separated with a semicolon (";")
  """
  sim_data = sim.data
  # Split quantities up by semicolon
  evals  = plot["quantity"].split(";") 
  if "qlabels" in plot:
    plot_labels = plot["qlabels"].split(";")
    for label in plot_labels:
      # Clean up labels
      label = label.strip()
  else:
    plot_labels = None
  plot_data = []
  for ev in evals:
    eval_data = sim_data.eval(ev)
    plot_data.append(eval_data)
  # Quick check to make sure that all labels are correctly applied
  # Labels can be left blank, but if labels are included, need one for each
  if plot_labels != None:
    if len(plot_data) != len(plot_labels):
      print("!!! Mismatch in number of labels and number of comparative plots")
      sys.exit()
  return plot_data,plot_labels

def defineParameter(dictionary, key, fallback = None):
  """
  Define a variable if it is contained in the dictionary, if not use a fallback
  Fairly self explanitory, this gets used a *lot*
  """
  if key in dictionary:
    parameter = dictionary[key]
  else:
    parameter = fallback
  return parameter

def main(**kwargs):
  # Load entire config file into memory
  config_filename = kwargs["config_file"]
  with open(config_filename) as config_file:
    config = yaml.load(config_file,Loader=yaml.FullLoader)
  # Read pgen files for each simulation, read in data for each simulation
  pgen_filenames = kwargs["pgen_files"]
  sims = []
  for filename in pgen_filenames:
    # Each simulation stored in the form of a class containing data and headers
    sims.append(Simulation(filename))
  # Configure global plotting parameters (figure size, etc.)
  plotheight = defineParameter(config,"plotheight",6)
  plotwidth  = defineParameter(config,"plotwidth",5)
  dpi        = defineParameter(config,"dpi",150)
  extension  = defineParameter(config,"extension","png")
  # Enable LaTeX plotting backend, requires TeXLive to be installed
  plt.rcParams.update({
      "text.usetex": True,
      "font.family": "serif",
  })
  # Process each individiual plot, overlaying each one
  for plot in config["plots"]:
    # Generate filename
    filename = ""
    # Add directory to filename
    if kwargs["dir"] != None:
      if os.path.exists(kwargs["dir"]):
        filename += kwargs["dir"]+"/"
      else:
        print("Directory does not exist!")
        sys.exit()
    # Add prefix to filename
    if kwargs["prefix"] != None:
        filename += kwargs["prefix"] + "."
    # Add quantity name to filename, or use custom name
    filename += defineParameter(plot,"filename",plot["quantity"])
    # Add extension to filename
    filename += "." + defineParameter(config,"extension","png")
    # Generate axes labels
    # X label
    plot_xlabel = config["xlabel"]
    xunit = defineParameter(config,"xunit")
    if xunit != None:
      plot_xlabel = "{0} ({1})".format(plot_xlabel,xunit)
    # Y label
    plot_ylabel = plot["ylabel"]
    yunit = defineParameter(plot,"yunit")
    if yunit != None:
      plot_ylabel = "{0} ({1})".format(plot_ylabel,yunit)
    # Establish plotting environment
    plt.figure(figsize=(plotwidth,plotheight),dpi=dpi)
    plt.grid(True,which="both",ls="dotted")
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    # Set log scale if needed
    islog = defineParameter(plot,"log",False)
    if islog:
      plt.yscale("log")
    # Loop through all sims, plotting all data
    for sim in sims:
      # Read data for particular sim
      plot_ydata,plot_labels = evalQuantity(plot,sim)
      # Get x axis data
      xquantity  = defineParameter(config,"xquantity","time")
      plot_xdata = sim.data[xquantity]
      # Scale x and y axes
      xscale = defineParameter(config,"xscale",1.0)
      yscale = defineParameter(plot,"yscale",1.0)
      # Plot data, two different plot calls depending on whether plots are overlaid or not
      for n in range(len(plot_ydata)):
        if plot_labels != None:
          plt.plot(plot_xdata*xscale,plot_ydata[n]*yscale,label=plot_labels[n])
        else:
          plt.plot(plot_xdata*xscale,plot_ydata[n]*yscale,label=sim.label)
    # Finish up figure and save
    plt.legend()
    plt.savefig(filename,
                bbox_inches="tight")
    # Confirmation that plot has finished
    print("> Finished {}".format(plot["filename"]))
  return


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Process 2D and hst data from a folder of Athena outputs.\nPositional arguments and flags can  be used, if all arguments are left blank, a config file will instead be used.")
  parser.add_argument("config_file",
                       help="name of yaml config file to be used.")
  parser.add_argument("pgen_files",
                      nargs="*",
                      help="Filenames of pgen files, if left blank, program will attempt to find any history files in current folder.")
  parser.add_argument("-p",
                      "--prefix",
                      type=str,
                      help="Add prefix to filename for plot outputs, by default does not use a prefix, prefix ammended in the form of foo.bar.ext, by default bar.ext where bar is the plot name derived from the config file.")
  parser.add_argument("-d",
                      "--dir",
                      type=str,
                      help="Save to directory, by default uses current working directory.")
  args = parser.parse_args()
  main(**vars(args))
