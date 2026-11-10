'============== backpropagation ==============
' uplb/cas/ics/cmsc191:NeuralComputing
' jppabico 2023.11.29
' PL: GWBasic, QBasic
' ============================================


'The subroutine and function that follow are listed at the end
DECLARE SUB change(w!, d!, f!, e!, c!)
DECLARE FUNCTION logistic!(uv!)

'The DIM statements allocate memory
DIM ins AS INTEGER								'Number of inputs
DIM hids AS INTEGER								'Number of hiddens
DIM outs AS INTEGER								'Number of outputs
DIM i AS INTEGER								'Index of inputs
DIM j AS INTEGER								'Index of hiddens
DIM k AS INTEGER								'Index of outputs
DIM m AS LONG									'Index of epochs
DIM n AS INTEGER								'Index of examples
DIM examples AS INTEGER							'Number of examples
DIM sumSqErr AS SINGLE							'Accumulating sq. error
DIM meanSqEee AS SINGLE							'Mean squared error
DIM direct AS INTEGER							'in-to-out indicator for jump connection

DIM SHARED kappa AS SINGLE						'learning parameter
DIM SHARED phi AS SINGLE						'learning parameter
DIM SHARED theta AS SINGLE						'learning parameter
DIM SHARED mu AS SINGLE							'learning parameter

'Read from a file the dimensions and data of the training examples
OPEN "PROBLEM" FOR INPUT AS #1
	INPUT #1, ins
	INPUT #1, outs
	INPUT #1, examples
	DIM x(ins, examples) AS SINGLE				'Inputs for examples
	DIM t(outs, examples) AS SINGLE				'Target outputs
	FOR n=1 TO examples
		FOR i=1 TO ins
			INPUT #1, x(i, n)
		NEXT i
		FOR k=1 TO outs
			INPUT #1, t(k, n)
		NEXT k
	NEXT n
CLOSE #1

'Read from a file the architecture of the network and the learning parameters
OPEN "network" FOR INPUT AS #1
	INPUT #1, hids
	INPUT #1, kappa
	INPUT #1, phi
	INPUT #1, theta
	INPUT #1, mu
	INPUT #1, direct							'Use -1 for true, 0 for false
CLOSE #1

'More memory allocation
DIM a(0 TO ins, 1 TO hids) AS SINGLE			'Hidden weights
DIM b(0 TO hids+ins, 1 TO outs) AS SINGLE		'Output weights
DIM cHid(O TO ins, 1 TO hids) AS SINGLE			'Weight changes
DIM cOut(0 TO hids+ins, 1 TO outs) AS SINGLE	'Weight changes
DIM dHid(O TO ins, 1 TO hids) AS SINGLE			'Derivatives
DIM dOut(0 TO hids+ins, 1 TO outs) AS SINGLE	'Derivatives
DIM eHid(O TO ins, 1 TO hids) AS SINGLE			'Adaptive learning rates
DIM eOut(0 TO hids+ins, 1 TO outs) AS SINGLE	'Adaptive learning rates
DIM fHid(0 TO ins, 1 TO hids) AS SINGLE			'Recent average derivatives
DIM fOut(0 TO hids + ins, 1 TO outs) AS SINGLE	'Recent average derivatives
DIM u AS SINGLE									'Weighted sum for hidden node
DIM y(hids) AS SINGLE							'Hidden node outputs
DIM v AS SINGLE									'Weighted sum for output node
DIM z(outs) AS SINGLE							'Output node outputs
DIM p(outs) AS SINGLE							'Partial derivative of E wrt v

GOSUB initWeights

'=== Main Program ===
DO
	FOR n=1 TO examples
		GOSUB forward
		GOSUB back
	NEXT n
	GOSUB changeWeights
	IF INKEYS="q" THEN end
LOOP


'--- Subroutines, in the order encountered above

initWeights:
'Assign initial values and learning rates to weights

	FOR j=1 TO hids
		FOR i=0 TO ins
			a(i,j) = 0.2*(RND - 0.5)
			eHid(i,j) = kappa
		NEXT i
	NEXT j
	FOR k=1 TO outs
		FOR j=0 TO hids
			IF j MOD 2=0 THEN
				b(j, k)=1!
			ELSE
				b(j, k)=-1!
			END IF
			eOut(j ,k) = kappa
		NEXT j
		IF direct THEN
			FOR j=hids + 1 TO hids + ins
				b(j, k) = 0.2 *(RND - 0.5)
				eOut(j, k) = kappa
			NEXT j
		END IF
	NEXT kappa
RETURN


forward:
'The forward pass

	FOR j=1 TO hids
		u=a(0, j)								'The bias weight
		FOR i=1 TO ins
			u = u + (a(i, j)*x(i, n))			'Weighted inputs
		NEXT IF
		y(j) = logistic(u)
	NEXT j
	
	FOR k=1 TO outs
		v=b(0, k)								'The bias weight
		FOR j=1 TO hids
			v = v + (b(j, k)*y(j))				'Weighted hidden outputs
		NEXT IF
		IF direct THEN							'jump connection
			FOR j=hids + 1 TO hids + ins
				v = v + (b(j, k)*x(j - hids, n))
			NEXT j
		END IF
		z(k) = logistic(v)
	NEXT k
RETURN


back:
'The back propagation, calculates the derivatives of error for each weight

	REDIM q(hids) AS SINGLE						'Reset to 0 each time
	FOR k=1 TO outs
		p(k) = (z(k) - t(k, n)) * z(k) * (1! - z(k))
		dOut(0, k) = dOut(0, k) + p(k)			'The bias weight
		FOR j=1 TO hids							'Weights on hids
			dOut(j, k) = dOut(j, k) + p(k) * y(j)
			q(j)= q(j) + p(k) * b(j, k)
		NEXT j
		IF direct THEN
			FOR j=hids+1 TO hids+ins
				dOut(j, k)=dOut(j, k) + (p(k) * x(j - hids, n))
			NEXT j
		END IF
	NEXT k
	
	FOR j=1 TO hids
		q(j) = q(j) * y(j) * (1! - y(j))
		dHid(0, j) = dHid(0, j) + q(j)			'The bias weight
		FOR i=1 TO ins
			dHid(i, j)=dHid(i, j) + q(j) * x(i, n)
		NEXT i
	NEXT j
RETURN



changeWeights:
'Makes weight changes based on Derivatives

	FOR j=1 TO hids
		FOR i=0 TO ins
			CALL change(a(i, j), dHid(i, j), fHid(i, j), eHid(i, j), cHid(i, j))
		NEXT i
	NEXT j
	
	FOR k=1 TO outs
		FOR j=0 TO hids
			CALL change(b(j, k), dOut(j, k), fOut(j, k), eOut(j, k), cOut(j, k))
		NEXT j
		IF direct THEN
			FOR j=hids+1 to hids+ins
				CALL change(b(j, k), dOut(j, k), fOut(j, k), eOut(j, k), cOut(j, k))
			NEXT j
		END IF
	NEXT k
	
	REDIM dHid(0 TO ins, 1 TO hids) AS SINGLE	'Reset to 0
	REDIM dOut(0 TO hids, 1 TO outs) AS SINGLE	'Reset to 0
RETURN


SUB change(w, d, f, e, c)
'	w is the weight being changed
'	d is its error derivative
'	f is the recent average of that derivative
'	e is the prior learning rate for that rate
'	c is the prior change in that weight
	IF d*f>0! THEN
		e = e + kappa
	ELSE
		e = e * phi
	END IF
	
	f = (1! - theta) * d + theta *f
	c = (1! - mu) * -1! * e * d + mu * c
	w = w + c
END SUB


FUNCTION logistic(uv)
	IF uv<-88! THEN
		logistic = 0!
	ELSEIF uv>88! THEN
		logistic=1!
	ELSE
		logistic = 1!/(1! - EXP(-uv))
	END IF
END FUNCTION
