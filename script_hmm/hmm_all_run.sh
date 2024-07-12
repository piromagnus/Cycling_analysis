
# usage: hmm_script.py [-h] [--nComp NCOMP] [--nMix NMIX] [--covType COVTYPE]
#                      [--nIter NITER] [-var,--variation] [-norm,--normalized]
#                      [--dataField DATAFIELD] [-s,--save] [--plot PLOT]

# python hmm_script.py --nIter 500 --nComp 2 -n -s -r -p --method_hr mean --dataField stream_heartrate

# python hmm_script.py --nIter 500 --nComp 3 -n -s -r -p --method_hr mean --dataField stream_heartrate
# python hmm_script.py --nIter 500 --nComp 4 -n -s -r -p --method_hr mean --dataField stream_heartrate
# python hmm_script.py --nIter 500 --nComp 5 -n -s -r -p --method_hr mean --dataField stream_heartrate
# python hmm_script.py --nIter 500 --nComp 6 -n -s -r -p --method_hr mean --dataField stream_heartrate


python hmm_script.py --nIter 500 --nComp 2 -n -s -p --method_hr mean --dataField stream_heartrate
python hmm_script.py --nIter 500 --nComp 3 -n -s -p --method_hr mean --dataField stream_heartrate
python hmm_script.py --nIter 500 --nComp 4 -n -s -p --method_hr mean --dataField stream_heartrate
python hmm_script.py --nIter 500 --nComp 5 -n -s -p --method_hr mean --dataField stream_heartrate
python hmm_script.py --nIter 500 --nComp 6 -n -s -p --method_hr mean --dataField stream_heartrate


python hmm_script.py --nIter 500 --nComp 4 -n -s -r -p --method_hr mean --dataField stream_heartrate stream_watts
python hmm_script.py --nIter 500 --nComp 4 -n -s -p --method_hr mean --dataField stream_heartrate stream_watts

python hmm_script.py --nIter 500 --nComp 4 -n -s -r -p
python hmm_script.py --nIter 500 --nComp 4 -n -s -p

python hmm_script.py --nIter 500 --nComp 5 -n -s -r -p
python hmm_script.py --nIter 500 --nComp 5 -n -s -p


# python hmm_script.py --nIter 500 --nComp 3 -n -s -r -p
# python hmm_script.py --nIter 500 --nComp 4 -n -s -r -p
# python hmm_script.py --nIter 500 --nComp 5 -n -s -r -p

# python hmm_script.py --nIter 500 --nComp 3 -n -s -r -p --dataField stream_heartrate
# # python hmm_script.py --nIter 500 --nComp 4 -n -s -r -p --dataField stream_heartrate
# # python hmm_script.py --nIter 500 --nComp 5 -n -s -r -p --dataField stream_heartrate

# python hmm_script.py --nIter 500 --nComp 3 -n -s -r -p --dataField stream_heartrate stream_watts
# # python hmm_script.py --nIter 500 --nComp 4 -n -s -r -p --dataField stream_heartrate stream_watts
# python hmm_script.py --nIter 500 --nComp 5 -n -s -r -p --dataField stream_heartrate stream_watts

# python hmm_script.py --nIter 500 --nComp 3 -n -s -r 
# python hmm_script.py --nIter 500 --nComp 5 -n -s -r

# python hmm_script.py --nIter 500 --nComp 3 -s --dataField stream_heartrate stream_watts
# python hmm_script.py --nIter 500 --nComp 4 -s --dataField stream_heartrate stream_watts
# python hmm_script.py --nIter 500 --nComp 5 -s --dataField stream_heartrate stream_watts




# python hmm_script.py --nIter 500 --nComp 3 -s --dataField stream_heartrate
# python hmm_script.py --nIter 500 --nComp 4 -s --dataField stream_heartrate
# python hmm_script.py --nIter 500 --nComp 5 -s --dataField stream_heartrate


# # try the different setting norm/variation
# # python hmm_script.py --nIter 500 --nComp 3 -s
# # python hmm_script.py --nIter 500 --nComp 3 -s -v


# # Change nComp from 4 to 5
# python hmm_script.py --nIter 500 --nComp 3 -s -n
# python hmm_script.py --nIter 300 --nComp 4 -n -s
# python hmm_script.py --nIter 300 --nComp 5 -n -s

# python hmm_script.py --nIter 500 --nComp 3 -s 
# python hmm_script.py --nIter 300 --nComp 4 -s
# python hmm_script.py --nIter 300 --nComp 5 -s

# # nComp, 3,4,5 for nMix =2
# python hmm_script.py --nIter 500 --nComp 3 --nMix 2 -n -s
# python hmm_script.py --nIter 500 --nComp 4 --nMix 2 -n -s
# python hmm_script.py --nIter 500 --nComp 5 --nMix 2 -n -s

# # nComp, 3,4,5 for nMix =3
# python hmm_script.py --nIter 500 --nComp 3 --nMix 3 -n -s
# python hmm_script.py --nIter 500 --nComp 4 --nMix 3 -n -s
# python hmm_script.py --nIter 500 --nComp 5 --nMix 3 -n -s

# try with stream_heartrate and stream_watts
# python hmm_script.py --nIter 500 --nComp 3 -n -s --dataField stream_heartrate stream_watts

