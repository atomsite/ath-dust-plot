"""
Plot files with gnuplot & PDFLaTeX
- Previously matplotlib was used to plot
- Requires the following packages
  - gnuplot
  - pdflatex
  - cairo
- On MacOS these can all be installed with brew
"""

def buildHeader(file,filename,size,palette):
  """
  Build the standard header
  carioLaTeX is used because it is more reliable and has better results
  """
  # Write header contents
  file.write("\# Automatically generated with athena_history_gnuplot.py")
  file.write("set terminal cairolatex size {}in,{}in crop standalone\n".format(size[0],size[1]))
  file.write("set output '{}'\n".format(filename))
  file.write("load '{}'\n".format(palette))
  # Finish and return
  return file
