#!/bin/bash
#
#Mouse myocyte model
# Compile
g++ Stats.cpp -o Stats
# Program Execution.
# The following will run the myocyte model using default parameters for the 6 ionic channel conductances, at a 250ms pacing interval, and return the eletrophysiological outputs of the model
echo "1.0 1.0 1.0 1.0 1.0 1.0" | ./Stats 250
# The following will output the ion channel currents during a voltage clamp to -10 mV  with double the default I_Ca,L conductance.
g++ Stats.cpp -o Stats -D___OUTPUT_CURRENTS
echo "2.0 1.0 1.0 1.0 1.0 1.0" | ./Stats 250 -10 2> VClamp_-10mV_currents.txt
#
#
#
#GES Search algorithm
# Compile
g++ downhill2.sh -o downhill2 
# Program Execution.
# The following will run the Nelder-Mead minimization algorithm 10,000 times, each time attempting to produce a GES parameter set from a random starting point in the parameter space. Solutions will be divided into 100 output files.
for i in {0..99}
do
./downhill2 $i
done
# Filter Non-physical and non-convering Solutions and extract GES parameter arrays
cat AMB5DRWF_5SP400_part* | gawk '($13 < 0.0025 && ($14-$15)**2 < 0.0001 && $19 < 130 && $7 > 0 && $8 > 0 && $9 > 0 && $10 > 0 && $11 > 0 && $12 > 0){print $7, $8, $9, $10, $11, $12}' > GES_results_filtered.txt

