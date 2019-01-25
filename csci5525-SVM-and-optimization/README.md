# csci5525-SVM-and-optimization
This is a course project.

Partner's name: Kai Wang

Instructions on how to run the code:

For question 1: 

$ python myDualSVM.py <dataFile> <regulatorC>

Then you will see the 10-fold validation result of this svm machine, which includes:
The average error rate
The standard deviation of the error rates

For question 2:

$ python myPegsos.py <dataFile> <batchSize k> <numberOfRuns>
$ python mySoftplus.py <dataFile> <batchSize k> <numberOfRuns>

Then you will see the performance of the gradient descent optimizer, which includes:
Average run time
Std of run time
Final loss decreasing rate
Final loss decreasing rate std

The maximum time to run the optimizers a single time is up to 200 seconds, which happens when batch size k == 2000.

The termination condition is when the total gradient times reaches 100n. So I also showed the final loss decreasing rate which is the decreasing rate when ktot reaches 100n, so as to compare the performance of different optimizers with different batch sizes.


