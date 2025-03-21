#!/bin/bash
############################################################
echo "Run this script on host machine window OUTSIDE docker"
############################################################
cp -rf matlab_plot/* ./
matlab << EOF
%%%%%%%%%%%%%%%%%%%%
%plot Fig. 1b
%Related script:
%figure1b_plot.m
%%%%%%%%%%%%%%%%%%%%
figure1b_plot

%%%%%%%%%%%%%%%%%%%%
%plot Fig. 2a
%Related script:
%figure2a_plot.m
%%%%%%%%%%%%%%%%%%%%
figure2a_plot

%%%%%%%%%%%%%%%%%%%%
%plot Fig. 2b
%Related script:
%figure2b_plot.m
%%%%%%%%%%%%%%%%%%%%
figure2b_plot

%%%%%%%%%%%%%%%%%%%%
%plot Fig. 2c
%Related script:
%figure2c_plot.m
%%%%%%%%%%%%%%%%%%%%
figure2c_plot

%%%%%%%%%%%%%%%%%%%%
%plot Fig. 2d
%Related script:
%figure2d_plot.m
%%%%%%%%%%%%%%%%%%%%
figure2d_plot

%%%%%%%%%%%%%%%%%%%%
%plot Fig. 2e
%Related script:
%figure2e_plot.m
%%%%%%%%%%%%%%%%%%%%
figure2e_plot

%%%%%%%%%%%%%%%%%%%%
%plot Fig. 2f
%Related script:
%figure2f_plot.m
%%%%%%%%%%%%%%%%%%%%
figure2f_plot

%%%%%%%%%%%%%%%%%%%%
%plot Fig. 4
%Related script:
%figure4a_plot.m
%%%%%%%%%%%%%%%%%%%%
figure4a_plot


%%%%%%%%%%%%%%%%%%%%
%plot Fig. 4
%Related script:
%figure4b_plot.m
%%%%%%%%%%%%%%%%%%%%
figure4b_plot

%%%%%%%%%%%%%%%%%%%%
%plot Fig. 4
%Related script:
%figure4c_plot.m
%%%%%%%%%%%%%%%%%%%%
figure4c_plot

%%%%%%%%%%%%%%%%%%%%
%plot Fig. 4
%Related script:
%figure4d_plot.m
%%%%%%%%%%%%%%%%%%%%
figure4d_plot

%%%%%%%%%%%%%%%%%%%%
%plot Fig. 4
%Related script:
%figure4e_plot.m
%%%%%%%%%%%%%%%%%%%%
figure4e_plot


%%%%%%%%%%%%%%%%%%%%
%plot Fig. 4
%Related script:
%figure4f_plot.m
%%%%%%%%%%%%%%%%%%%%
figure4f_plot

EOF

