#!/bin/sh
python IndenpendenceTest.py SVMcrossvalidationfirst_RFH.model SuppS4_output_RFH.csv;
python IndenpendenceTest.py SVMcrossvalidationfirst_PseEIIP.model SuppS4_output_pseEIIP.csv;
python IndenpendenceTest.py SVMcrossvalidationfirst_pseDNC.model SuppS4_output_pseDNC.csv;
python IndenpendenceTest.py SVMcrossvalidationfirst_PCP.model SuppS4_output_PCP.csv;
python IndenpendenceTest.py SVMcrossvalidationfirst_KNN.model SuppS4_output_KNN.csv;
python IndenpendenceTest.py SVMcrossvalidationfirst_Ath.model SuppS4_output_Ath.csv;
python IndenpendenceTest.py SVMcrossvalidationfirst_3mer.model SuppS4_output_3mr.csv;
python IndenpendenceTest.py SVMcrossvalidationfirst_2-4mer.model SuppS4_output_2-4mer.csv;
python IndenpendenceTest.py SVMcrossvalidationfirst_pseAAC.model SuppS4_output_pseAAC.csv;
python IndenpendenceTest.py SVMcrossvalidationfirst_dwt_rbio3.model SuppS4_output_dwt.csv;
python IndenpendenceTest.py SVMcrossvalidationfirst_MMI.model SuppS4_output_MMI.csv;