import re
import numpy
def ReadOutput(filnm,word):
#word='eigenvalue:'
  carray=numpy.empty(800,dtype=numpy.complex128)
  numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
  rx = re.compile(numeric_const_pattern, re.VERBOSE)
  print(rx.findall("Some example: Jr. it. was .23 between 2.3 and 42.31 seconds"))
  with open(filnm, 'r') as fp:
    # read all lines in a list
    lines = fp.readlines()
    nr_lines=0
    for line in lines:
        # check if string present on a current line
        if line.find(word) != -1:
            print(word, 'string exists in file')
            print('Line Number:', lines.index(line))
            print('Line:', line)
            print(rx.findall(line))
            fl_line=rx.findall(line)
            carray[int(fl_line[0])-1]=complex(float(fl_line[1]),float(fl_line[2]))
            nr_lines=nr_lines+1
  print(carray[:nr_lines])
  return carray[:nr_lines]
#filnm='Eigenv.out'
#word='eigenvalue:'
#ReadOutput(word,filnm)
