import athenaplotlib as apl
import argparse

def main():
  time = float(apl.)

  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Process 2D and hst data from a folder of Athena outputs.\nRequires a config file and a positional argument of at least one config file, additional prefixes and directories can also be added")
  parser.add_argument("pgen_filename",
                      help="Name of pgen file used for this program")
  parser.add_argument("phase",
                      type=float,
                      help="Phase to find, program finds first phase larger than this number, since value may not be exact")
