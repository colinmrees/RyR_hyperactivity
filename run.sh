#!/bin/bash
#
#Rabbit myocyte model
# Compile
make il
#Program execution
Device=0 #GPU Device
#Stabilized parameters
./cuda_Cell_Detailed $Device Rabbit_lqt2_stab 1.0 740 1.0 10 0.00 1.0 0.7 1 > /dev/null &
#Hyperactivity parameters
./cuda_Cell_Detailed $Device Rabbit_lqt2_hyper 1.0 740 1.0 3.333 0.00 18. 0.7 1 > /dev/null &
