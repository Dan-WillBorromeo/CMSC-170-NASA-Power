#=======================================================================
# PERL-style pseudo code for some subroutines realized from CMSC 191 lessons
#
# FUNCTION
#	logistic
#
# SUBROUTINES
#	forward
#	backward
#
# cc2023 /uplb/ics/jppabico, 2023.11.09
#=======================================================================


#-----------------------------------------------------------------------
# Function: $logval=logistic($uv)
# Goal: Calculates the logistic function of $uv
# 
# Parameter:
#	$uv = Any number
#
# Return:
#	$logval = the logistic of $uv
#-----------------------------------------------------------------------
sub logistic($uv) {
	my $logval=0;
	if ($uv<-88) {
		$logval = 0;	
	}
	elsif ($uv>88) {
		$logval = 1;
	}
	else {
		$logval = 1 / (1 + exp(-$uv));
	}
	return $logval;
}

#-----------------------------------------------------------------------
# Subroutine: forward
# Goal: Implements the forward pass
#
# Parameters:
#	@x = two-dimensional array of $numInputs inputs with $n examples
#	@a = 2D array of weights for the hidden nodes
#	@b = 2D array of weights for the output nodes
#	@z = a vector of $numOutputs outputs
#	$numInputs = number of input nodes
#	$numHidden = number of hidden nodes
#	$numOutputs= number of output nodes
#	$n = index of the example
#	$jump = 1 if the ANN is jump-connected, 0 otherwise
#
# Note: Certainly, the arrays can be passed by reference rather than by value to save on memory
#-----------------------------------------------------------------------
sub forward(@x, @a, @b, @z, $numInputs, $numHidden, $numOutputs, $n, $jump) {
	my $i, $j, $k, $u, $v;
	
	#forward pass from input layer to hidden layer
	for ($j=1; $j<=$numHidden; $j++) {
		$u = $a[0][$j];							#initialize u with the bias weight of the jth hidden node

		for ($i=; $i<=$numInputs; $i++) {
			$u = $u + $a[$i][$j] * $x[$i][$n];	#weighted inputs for the jth hidden node
		}

		$y[$j] = logistic($u);
	}
	
	#forward pass from hidden layer to output layer
	for ($k=1; $k<=numOutputs; $i++) {
		$v = $b[0][$k];							#initialize v with the bias weight of the kth output node
		
		for ($j=1; $j<=$numHidden; $j++) {
			$v = $v + $b[$j][$k] * $y[$j];		#weighted inputs for the kth output node
		}
		
		#if the ANN is jump connected
		if ($jump) {
			for ($j=$numHidden+1; $j<=$numHidden+$numInputs; $j++) {
				$v = $v + $b[$j][$k]*$x[$j-$numHidden][$n];
			}
		}
		
		$z[$k] = logistic($v);
		
	}

}

#-----------------------------------------------------------------------
# Subroutine: backward 
# Goal: Calculates the derivatives of error for each weight
#
# Parameters:
#	@x = two-dimensional array of $numInputs inputs with $n examples
#	@a = 2D array of weights for the hidden nodes
#	@b = 2D array of weights for the output nodes
#	@z = a vector of $numOutputs outputs
#	@t = two-dimensional array of $numOutputs target outputs with $n examples
#	$numInputs = number of input nodes
#	$numHidden = number of hidden nodes
#	$numOutputs= number of output nodes
#	$n = index of the example
#	$jump = 1 if the ANN is jump-connected, 0 otherwise
#
# Note: see above
#-----------------------------------------------------------------------
sub backward(@x, @a, @b, @z, @t, $numInputs, $numHidden, $numOutputs, $n, $jump) {
	my @q = (0) x $numHidden; 	#create a vector @q of size $numHidden and initialize the elements to zeroes
								#@q will hold the partial derivatives of E w.r.t. u
								
	for ($k=1; $l<= $numOutputs; $i++) {

		$p[$k] = ($z[$k] - $t[$k][$n]) * $z[$k] * (1 - $z[$k]);
		$dOut[0][$k] = $dOut[0][$k] + $p[$k];	#derivatives of the bias weight

		for ($j=1; $j<= $numHidden; $j++) {
			$dOut[$j][$k] = $dOut[$j][$k] + $p[$k] * $y[$j];
			$q[$j] = $q[$j] + $p[$k] * $b[$j][$k];
		}	
		if ($jump) {
			for ($j=$numHidden; $j<=$numHidden+$numInputs; $i++) {
				$dOut[$j][$k] = $dOut[$j][$k] + $p[$k] * $x[$j - $numHidden][$n];
			}
		}
	}
	
	for ($j=1; $j<=$numHidden; $j++) {
		$q[$j] = $q[$j] * $y[$j] * (1 - $y[$j]);
		$dHid[0][$j] = $dHid[0][$j] + $q[$j];
		
		for ($i=1; $i<=$numInputs; $i++) {
			$dHid[$i][$j] = $dHid[$i][$j] + $q[$j] * $x[$j][$n];
		}
	}
}
