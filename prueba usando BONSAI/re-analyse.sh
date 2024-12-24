#!/bin/sh

echo 'Analysing process started... This can take some time. Go and do something useful!\n\n';
path_to_starfinder="../../.."

echo 'Analysing the model density:';
echo '----------------------------';
perl $path_to_starfinder/analyse-model-density.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --mquantities 1 LOGL --mode analyse --path $path_to_starfinder --re-analyse 1 

echo '\n';

echo 'Analysing the posteriorpredictive checks:';
echo '-----------------------------------------';
perl $path_to_starfinder/analyse-posterior-predictive-checks.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --mquantities 1 LOGL --path $path_to_starfinder --re-analyse 1 

echo '\n';

echo 'Analysing parameter space coverage:';
echo '-----------------------------------';
perl $path_to_starfinder/analyse-parameter-space-coverage.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --path $path_to_starfinder --re-analyse 1 

echo '\n';

echo 'Analysing the histograms:';
echo '-------------------------';
perl $path_to_starfinder/analyse-histogram.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key AGE --results-file results.dat --path $path_to_starfinder --re-analyse 1 
perl $path_to_starfinder/analyse-histogram.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key C --results-file results.dat --path $path_to_starfinder --re-analyse 1 
perl $path_to_starfinder/analyse-histogram.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key HE --results-file results.dat --path $path_to_starfinder --re-analyse 1 
perl $path_to_starfinder/analyse-histogram.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key LOGG --results-file results.dat --path $path_to_starfinder --re-analyse 1 
perl $path_to_starfinder/analyse-histogram.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key LOGL --results-file results.dat --path $path_to_starfinder --re-analyse 1 
perl $path_to_starfinder/analyse-histogram.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key MACT --results-file results.dat --path $path_to_starfinder --re-analyse 1 
perl $path_to_starfinder/analyse-histogram.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key MINI --results-file results.dat --path $path_to_starfinder --re-analyse 1 
perl $path_to_starfinder/analyse-histogram.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key N --results-file results.dat --path $path_to_starfinder --re-analyse 1 
perl $path_to_starfinder/analyse-histogram.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key O --results-file results.dat --path $path_to_starfinder --re-analyse 1 
perl $path_to_starfinder/analyse-histogram.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key R --results-file results.dat --path $path_to_starfinder --re-analyse 1 
perl $path_to_starfinder/analyse-histogram.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key TEFF --results-file results.dat --path $path_to_starfinder --re-analyse 1 
perl $path_to_starfinder/analyse-histogram.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key VROT --results-file results.dat --path $path_to_starfinder --re-analyse 1 
perl $path_to_starfinder/analyse-histogram.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key VROTINI --results-file results.dat --path $path_to_starfinder --re-analyse 1 
perl $path_to_starfinder/analyse-histogram.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key XHE --results-file results.dat --path $path_to_starfinder --re-analyse 1 

echo '\n';

echo 'Analysing the error ellipses:';
echo '-----------------------------';
perl $path_to_starfinder/analyse-error-ellipse.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key AGE-MINI --contours 0 --path $path_to_starfinder --re-analyse 1 
perl $path_to_starfinder/analyse-error-ellipse.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key AGE-VROTINI --contours 0 --path $path_to_starfinder --re-analyse 1 
perl $path_to_starfinder/analyse-error-ellipse.pl --output-path ./output/ --starid 110277 --starname BS-06001204 --grid BONN-LMC --key MINI-VROTINI --contours 0 --path $path_to_starfinder --re-analyse 1 

echo '\n\nEverything done. :-)\n';
