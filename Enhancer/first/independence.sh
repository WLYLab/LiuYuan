#!/bin/sh
python RFH.py new_test.txt new_test_RFH.csv DNA 0;
python AthMethPre.py new_test.txt new_test_Ath.csv DNA 0;
python AthMethPre_2-4mer.py new_test.txt new_test_2-4mer.csv DNA 0;
python KNN.py new_test.txt new_test_KNN.csv DNA 0;
python PCP.py new_test.txt new_test_PCP.csv DNA 0;
python PseDNC.py new_test.txt new_test_pseDNC.csv DNA 0;
python PseEIIP.py new_test.txt new_test_pseEIIP.csv DNA 0;