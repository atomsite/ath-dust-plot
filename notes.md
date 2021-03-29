# Notes

## Athena Plot

## Athena Hist

- Histogram plotter currently reads in data all correctly, but the issue is parsing:
- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.eval.html
- Look into this, this might be the fix I need!
- will need to rewrite the plots section in the yaml:
- have an optional `type` flag:
  - `compare` plots multiple datasets on top of each other
  - `eval`    performs the pandas evaluation in a string in the `eval` variable
    - needs to be correctly done, but this means it works in line with no parsing needed
  - plotseparately will plot each and not overlay plots, default false