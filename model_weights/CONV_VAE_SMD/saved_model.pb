ɩ
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
3
Square
x"T
y"T"
Ttype:
2
	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
!autoencoder/encoder/conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*2
shared_name#!autoencoder/encoder/conv1d/kernel
?
5autoencoder/encoder/conv1d/kernel/Read/ReadVariableOpReadVariableOp!autoencoder/encoder/conv1d/kernel*"
_output_shapes
:&*
dtype0
?
autoencoder/encoder/conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!autoencoder/encoder/conv1d/bias
?
3autoencoder/encoder/conv1d/bias/Read/ReadVariableOpReadVariableOpautoencoder/encoder/conv1d/bias*
_output_shapes
:*
dtype0
?
#autoencoder/encoder/conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#autoencoder/encoder/conv1d_1/kernel
?
7autoencoder/encoder/conv1d_1/kernel/Read/ReadVariableOpReadVariableOp#autoencoder/encoder/conv1d_1/kernel*"
_output_shapes
:*
dtype0
?
!autoencoder/encoder/conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!autoencoder/encoder/conv1d_1/bias
?
5autoencoder/encoder/conv1d_1/bias/Read/ReadVariableOpReadVariableOp!autoencoder/encoder/conv1d_1/bias*
_output_shapes
:*
dtype0
?
 autoencoder/encoder/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" autoencoder/encoder/dense/kernel
?
4autoencoder/encoder/dense/kernel/Read/ReadVariableOpReadVariableOp autoencoder/encoder/dense/kernel*
_output_shapes
:	?*
dtype0
?
autoencoder/encoder/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name autoencoder/encoder/dense/bias
?
2autoencoder/encoder/dense/bias/Read/ReadVariableOpReadVariableOpautoencoder/encoder/dense/bias*
_output_shapes
:*
dtype0
?
"autoencoder/encoder/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"autoencoder/encoder/dense_1/kernel
?
6autoencoder/encoder/dense_1/kernel/Read/ReadVariableOpReadVariableOp"autoencoder/encoder/dense_1/kernel*
_output_shapes
:	?*
dtype0
?
 autoencoder/encoder/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" autoencoder/encoder/dense_1/bias
?
4autoencoder/encoder/dense_1/bias/Read/ReadVariableOpReadVariableOp autoencoder/encoder/dense_1/bias*
_output_shapes
:*
dtype0
?
"autoencoder/decoder/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"autoencoder/decoder/dense_2/kernel
?
6autoencoder/decoder/dense_2/kernel/Read/ReadVariableOpReadVariableOp"autoencoder/decoder/dense_2/kernel*
_output_shapes
:	?*
dtype0
?
 autoencoder/decoder/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" autoencoder/decoder/dense_2/bias
?
4autoencoder/decoder/dense_2/bias/Read/ReadVariableOpReadVariableOp autoencoder/decoder/dense_2/bias*
_output_shapes	
:?*
dtype0
?
+autoencoder/decoder/conv1d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+autoencoder/decoder/conv1d_transpose/kernel
?
?autoencoder/decoder/conv1d_transpose/kernel/Read/ReadVariableOpReadVariableOp+autoencoder/decoder/conv1d_transpose/kernel*"
_output_shapes
:*
dtype0
?
)autoencoder/decoder/conv1d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)autoencoder/decoder/conv1d_transpose/bias
?
=autoencoder/decoder/conv1d_transpose/bias/Read/ReadVariableOpReadVariableOp)autoencoder/decoder/conv1d_transpose/bias*
_output_shapes
:*
dtype0
?
-autoencoder/decoder/conv1d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*>
shared_name/-autoencoder/decoder/conv1d_transpose_1/kernel
?
Aautoencoder/decoder/conv1d_transpose_1/kernel/Read/ReadVariableOpReadVariableOp-autoencoder/decoder/conv1d_transpose_1/kernel*"
_output_shapes
:&*
dtype0
?
+autoencoder/decoder/conv1d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*<
shared_name-+autoencoder/decoder/conv1d_transpose_1/bias
?
?autoencoder/decoder/conv1d_transpose_1/bias/Read/ReadVariableOpReadVariableOp+autoencoder/decoder/conv1d_transpose_1/bias*
_output_shapes
:&*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
(Adam/autoencoder/encoder/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*9
shared_name*(Adam/autoencoder/encoder/conv1d/kernel/m
?
<Adam/autoencoder/encoder/conv1d/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/autoencoder/encoder/conv1d/kernel/m*"
_output_shapes
:&*
dtype0
?
&Adam/autoencoder/encoder/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/autoencoder/encoder/conv1d/bias/m
?
:Adam/autoencoder/encoder/conv1d/bias/m/Read/ReadVariableOpReadVariableOp&Adam/autoencoder/encoder/conv1d/bias/m*
_output_shapes
:*
dtype0
?
*Adam/autoencoder/encoder/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/autoencoder/encoder/conv1d_1/kernel/m
?
>Adam/autoencoder/encoder/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/autoencoder/encoder/conv1d_1/kernel/m*"
_output_shapes
:*
dtype0
?
(Adam/autoencoder/encoder/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/autoencoder/encoder/conv1d_1/bias/m
?
<Adam/autoencoder/encoder/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOp(Adam/autoencoder/encoder/conv1d_1/bias/m*
_output_shapes
:*
dtype0
?
'Adam/autoencoder/encoder/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*8
shared_name)'Adam/autoencoder/encoder/dense/kernel/m
?
;Adam/autoencoder/encoder/dense/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense/kernel/m*
_output_shapes
:	?*
dtype0
?
%Adam/autoencoder/encoder/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/autoencoder/encoder/dense/bias/m
?
9Adam/autoencoder/encoder/dense/bias/m/Read/ReadVariableOpReadVariableOp%Adam/autoencoder/encoder/dense/bias/m*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/encoder/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*:
shared_name+)Adam/autoencoder/encoder/dense_1/kernel/m
?
=Adam/autoencoder/encoder/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/encoder/dense_1/kernel/m*
_output_shapes
:	?*
dtype0
?
'Adam/autoencoder/encoder/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/autoencoder/encoder/dense_1/bias/m
?
;Adam/autoencoder/encoder/dense_1/bias/m/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense_1/bias/m*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/decoder/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*:
shared_name+)Adam/autoencoder/decoder/dense_2/kernel/m
?
=Adam/autoencoder/decoder/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/decoder/dense_2/kernel/m*
_output_shapes
:	?*
dtype0
?
'Adam/autoencoder/decoder/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'Adam/autoencoder/decoder/dense_2/bias/m
?
;Adam/autoencoder/decoder/dense_2/bias/m/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/decoder/dense_2/bias/m*
_output_shapes	
:?*
dtype0
?
2Adam/autoencoder/decoder/conv1d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42Adam/autoencoder/decoder/conv1d_transpose/kernel/m
?
FAdam/autoencoder/decoder/conv1d_transpose/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/autoencoder/decoder/conv1d_transpose/kernel/m*"
_output_shapes
:*
dtype0
?
0Adam/autoencoder/decoder/conv1d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adam/autoencoder/decoder/conv1d_transpose/bias/m
?
DAdam/autoencoder/decoder/conv1d_transpose/bias/m/Read/ReadVariableOpReadVariableOp0Adam/autoencoder/decoder/conv1d_transpose/bias/m*
_output_shapes
:*
dtype0
?
4Adam/autoencoder/decoder/conv1d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*E
shared_name64Adam/autoencoder/decoder/conv1d_transpose_1/kernel/m
?
HAdam/autoencoder/decoder/conv1d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp4Adam/autoencoder/decoder/conv1d_transpose_1/kernel/m*"
_output_shapes
:&*
dtype0
?
2Adam/autoencoder/decoder/conv1d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*C
shared_name42Adam/autoencoder/decoder/conv1d_transpose_1/bias/m
?
FAdam/autoencoder/decoder/conv1d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOp2Adam/autoencoder/decoder/conv1d_transpose_1/bias/m*
_output_shapes
:&*
dtype0
?
(Adam/autoencoder/encoder/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*9
shared_name*(Adam/autoencoder/encoder/conv1d/kernel/v
?
<Adam/autoencoder/encoder/conv1d/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/autoencoder/encoder/conv1d/kernel/v*"
_output_shapes
:&*
dtype0
?
&Adam/autoencoder/encoder/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/autoencoder/encoder/conv1d/bias/v
?
:Adam/autoencoder/encoder/conv1d/bias/v/Read/ReadVariableOpReadVariableOp&Adam/autoencoder/encoder/conv1d/bias/v*
_output_shapes
:*
dtype0
?
*Adam/autoencoder/encoder/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/autoencoder/encoder/conv1d_1/kernel/v
?
>Adam/autoencoder/encoder/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/autoencoder/encoder/conv1d_1/kernel/v*"
_output_shapes
:*
dtype0
?
(Adam/autoencoder/encoder/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/autoencoder/encoder/conv1d_1/bias/v
?
<Adam/autoencoder/encoder/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOp(Adam/autoencoder/encoder/conv1d_1/bias/v*
_output_shapes
:*
dtype0
?
'Adam/autoencoder/encoder/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*8
shared_name)'Adam/autoencoder/encoder/dense/kernel/v
?
;Adam/autoencoder/encoder/dense/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense/kernel/v*
_output_shapes
:	?*
dtype0
?
%Adam/autoencoder/encoder/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/autoencoder/encoder/dense/bias/v
?
9Adam/autoencoder/encoder/dense/bias/v/Read/ReadVariableOpReadVariableOp%Adam/autoencoder/encoder/dense/bias/v*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/encoder/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*:
shared_name+)Adam/autoencoder/encoder/dense_1/kernel/v
?
=Adam/autoencoder/encoder/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/encoder/dense_1/kernel/v*
_output_shapes
:	?*
dtype0
?
'Adam/autoencoder/encoder/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/autoencoder/encoder/dense_1/bias/v
?
;Adam/autoencoder/encoder/dense_1/bias/v/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/encoder/dense_1/bias/v*
_output_shapes
:*
dtype0
?
)Adam/autoencoder/decoder/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*:
shared_name+)Adam/autoencoder/decoder/dense_2/kernel/v
?
=Adam/autoencoder/decoder/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/autoencoder/decoder/dense_2/kernel/v*
_output_shapes
:	?*
dtype0
?
'Adam/autoencoder/decoder/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'Adam/autoencoder/decoder/dense_2/bias/v
?
;Adam/autoencoder/decoder/dense_2/bias/v/Read/ReadVariableOpReadVariableOp'Adam/autoencoder/decoder/dense_2/bias/v*
_output_shapes	
:?*
dtype0
?
2Adam/autoencoder/decoder/conv1d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42Adam/autoencoder/decoder/conv1d_transpose/kernel/v
?
FAdam/autoencoder/decoder/conv1d_transpose/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/autoencoder/decoder/conv1d_transpose/kernel/v*"
_output_shapes
:*
dtype0
?
0Adam/autoencoder/decoder/conv1d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adam/autoencoder/decoder/conv1d_transpose/bias/v
?
DAdam/autoencoder/decoder/conv1d_transpose/bias/v/Read/ReadVariableOpReadVariableOp0Adam/autoencoder/decoder/conv1d_transpose/bias/v*
_output_shapes
:*
dtype0
?
4Adam/autoencoder/decoder/conv1d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*E
shared_name64Adam/autoencoder/decoder/conv1d_transpose_1/kernel/v
?
HAdam/autoencoder/decoder/conv1d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp4Adam/autoencoder/decoder/conv1d_transpose_1/kernel/v*"
_output_shapes
:&*
dtype0
?
2Adam/autoencoder/decoder/conv1d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*C
shared_name42Adam/autoencoder/decoder/conv1d_transpose_1/bias/v
?
FAdam/autoencoder/decoder/conv1d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOp2Adam/autoencoder/decoder/conv1d_transpose_1/bias/v*
_output_shapes
:&*
dtype0

NoOpNoOp
?T
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?T
value?TB?T B?T
?
encoder
decoder
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?
		conv0
	
conv1
flat

dense_mean
dense_log_var
sampling
	variables
trainable_variables
regularization_losses
	keras_api
?
	dense
reshape
	conv1
	conv2
	variables
trainable_variables
regularization_losses
	keras_api
?
iter

beta_1

beta_2
	decay
learning_rate m?!m?"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?-m? v?!v?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?-v?
f
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
f
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
 
?
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
 
h

 kernel
!bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
h

"kernel
#bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
R
;	variables
<trainable_variables
=regularization_losses
>	keras_api
h

$kernel
%bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
h

&kernel
'bias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
R
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
8
 0
!1
"2
#3
$4
%5
&6
'7
8
 0
!1
"2
#3
$4
%5
&6
'7
 
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
h

(kernel
)bias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
R
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
h

*kernel
+bias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
h

,kernel
-bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
*
(0
)1
*2
+3
,4
-5
*
(0
)1
*2
+3
,4
-5
 
?
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!autoencoder/encoder/conv1d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEautoencoder/encoder/conv1d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#autoencoder/encoder/conv1d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!autoencoder/encoder/conv1d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE autoencoder/encoder/dense/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEautoencoder/encoder/dense/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"autoencoder/encoder/dense_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE autoencoder/encoder/dense_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"autoencoder/decoder/dense_2/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE autoencoder/decoder/dense_2/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+autoencoder/decoder/conv1d_transpose/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)autoencoder/decoder/conv1d_transpose/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-autoencoder/decoder/conv1d_transpose_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+autoencoder/decoder/conv1d_transpose_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

e0
 
 

 0
!1

 0
!1
 
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
3	variables
4trainable_variables
5regularization_losses

"0
#1

"0
#1
 
?
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
7	variables
8trainable_variables
9regularization_losses
 
 
 
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
;	variables
<trainable_variables
=regularization_losses

$0
%1

$0
%1
 
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
?	variables
@trainable_variables
Aregularization_losses

&0
'1

&0
'1
 
?
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
 
 
 
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
 
*
	0

1
2
3
4
5
 
 
 

(0
)1

(0
)1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses

*0
+1

*0
+1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses

,0
-1

,0
-1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
 

0
1
2
3
 
 
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
?~
VARIABLE_VALUE(Adam/autoencoder/encoder/conv1d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/autoencoder/encoder/conv1d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/autoencoder/encoder/conv1d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE(Adam/autoencoder/encoder/conv1d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/autoencoder/encoder/dense/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE%Adam/autoencoder/encoder/dense/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/autoencoder/encoder/dense_1/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/autoencoder/encoder/dense_1/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/autoencoder/decoder/dense_2/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/autoencoder/decoder/dense_2/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/autoencoder/decoder/conv1d_transpose/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/autoencoder/decoder/conv1d_transpose/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/autoencoder/decoder/conv1d_transpose_1/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/autoencoder/decoder/conv1d_transpose_1/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE(Adam/autoencoder/encoder/conv1d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/autoencoder/encoder/conv1d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/autoencoder/encoder/conv1d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE(Adam/autoencoder/encoder/conv1d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/autoencoder/encoder/dense/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE%Adam/autoencoder/encoder/dense/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/autoencoder/encoder/dense_1/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/autoencoder/encoder/dense_1/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE)Adam/autoencoder/decoder/dense_2/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/autoencoder/decoder/dense_2/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/autoencoder/decoder/conv1d_transpose/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/autoencoder/decoder/conv1d_transpose/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/autoencoder/decoder/conv1d_transpose_1/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/autoencoder/decoder/conv1d_transpose_1/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????d&*
dtype0* 
shape:?????????d&
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1!autoencoder/encoder/conv1d/kernelautoencoder/encoder/conv1d/bias#autoencoder/encoder/conv1d_1/kernel!autoencoder/encoder/conv1d_1/bias autoencoder/encoder/dense/kernelautoencoder/encoder/dense/bias"autoencoder/encoder/dense_1/kernel autoencoder/encoder/dense_1/bias"autoencoder/decoder/dense_2/kernel autoencoder/decoder/dense_2/bias+autoencoder/decoder/conv1d_transpose/kernel)autoencoder/decoder/conv1d_transpose/bias-autoencoder/decoder/conv1d_transpose_1/kernel+autoencoder/decoder/conv1d_transpose_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d&*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_27540
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp5autoencoder/encoder/conv1d/kernel/Read/ReadVariableOp3autoencoder/encoder/conv1d/bias/Read/ReadVariableOp7autoencoder/encoder/conv1d_1/kernel/Read/ReadVariableOp5autoencoder/encoder/conv1d_1/bias/Read/ReadVariableOp4autoencoder/encoder/dense/kernel/Read/ReadVariableOp2autoencoder/encoder/dense/bias/Read/ReadVariableOp6autoencoder/encoder/dense_1/kernel/Read/ReadVariableOp4autoencoder/encoder/dense_1/bias/Read/ReadVariableOp6autoencoder/decoder/dense_2/kernel/Read/ReadVariableOp4autoencoder/decoder/dense_2/bias/Read/ReadVariableOp?autoencoder/decoder/conv1d_transpose/kernel/Read/ReadVariableOp=autoencoder/decoder/conv1d_transpose/bias/Read/ReadVariableOpAautoencoder/decoder/conv1d_transpose_1/kernel/Read/ReadVariableOp?autoencoder/decoder/conv1d_transpose_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp<Adam/autoencoder/encoder/conv1d/kernel/m/Read/ReadVariableOp:Adam/autoencoder/encoder/conv1d/bias/m/Read/ReadVariableOp>Adam/autoencoder/encoder/conv1d_1/kernel/m/Read/ReadVariableOp<Adam/autoencoder/encoder/conv1d_1/bias/m/Read/ReadVariableOp;Adam/autoencoder/encoder/dense/kernel/m/Read/ReadVariableOp9Adam/autoencoder/encoder/dense/bias/m/Read/ReadVariableOp=Adam/autoencoder/encoder/dense_1/kernel/m/Read/ReadVariableOp;Adam/autoencoder/encoder/dense_1/bias/m/Read/ReadVariableOp=Adam/autoencoder/decoder/dense_2/kernel/m/Read/ReadVariableOp;Adam/autoencoder/decoder/dense_2/bias/m/Read/ReadVariableOpFAdam/autoencoder/decoder/conv1d_transpose/kernel/m/Read/ReadVariableOpDAdam/autoencoder/decoder/conv1d_transpose/bias/m/Read/ReadVariableOpHAdam/autoencoder/decoder/conv1d_transpose_1/kernel/m/Read/ReadVariableOpFAdam/autoencoder/decoder/conv1d_transpose_1/bias/m/Read/ReadVariableOp<Adam/autoencoder/encoder/conv1d/kernel/v/Read/ReadVariableOp:Adam/autoencoder/encoder/conv1d/bias/v/Read/ReadVariableOp>Adam/autoencoder/encoder/conv1d_1/kernel/v/Read/ReadVariableOp<Adam/autoencoder/encoder/conv1d_1/bias/v/Read/ReadVariableOp;Adam/autoencoder/encoder/dense/kernel/v/Read/ReadVariableOp9Adam/autoencoder/encoder/dense/bias/v/Read/ReadVariableOp=Adam/autoencoder/encoder/dense_1/kernel/v/Read/ReadVariableOp;Adam/autoencoder/encoder/dense_1/bias/v/Read/ReadVariableOp=Adam/autoencoder/decoder/dense_2/kernel/v/Read/ReadVariableOp;Adam/autoencoder/decoder/dense_2/bias/v/Read/ReadVariableOpFAdam/autoencoder/decoder/conv1d_transpose/kernel/v/Read/ReadVariableOpDAdam/autoencoder/decoder/conv1d_transpose/bias/v/Read/ReadVariableOpHAdam/autoencoder/decoder/conv1d_transpose_1/kernel/v/Read/ReadVariableOpFAdam/autoencoder/decoder/conv1d_transpose_1/bias/v/Read/ReadVariableOpConst*>
Tin7
523	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_28215
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate!autoencoder/encoder/conv1d/kernelautoencoder/encoder/conv1d/bias#autoencoder/encoder/conv1d_1/kernel!autoencoder/encoder/conv1d_1/bias autoencoder/encoder/dense/kernelautoencoder/encoder/dense/bias"autoencoder/encoder/dense_1/kernel autoencoder/encoder/dense_1/bias"autoencoder/decoder/dense_2/kernel autoencoder/decoder/dense_2/bias+autoencoder/decoder/conv1d_transpose/kernel)autoencoder/decoder/conv1d_transpose/bias-autoencoder/decoder/conv1d_transpose_1/kernel+autoencoder/decoder/conv1d_transpose_1/biastotalcount(Adam/autoencoder/encoder/conv1d/kernel/m&Adam/autoencoder/encoder/conv1d/bias/m*Adam/autoencoder/encoder/conv1d_1/kernel/m(Adam/autoencoder/encoder/conv1d_1/bias/m'Adam/autoencoder/encoder/dense/kernel/m%Adam/autoencoder/encoder/dense/bias/m)Adam/autoencoder/encoder/dense_1/kernel/m'Adam/autoencoder/encoder/dense_1/bias/m)Adam/autoencoder/decoder/dense_2/kernel/m'Adam/autoencoder/decoder/dense_2/bias/m2Adam/autoencoder/decoder/conv1d_transpose/kernel/m0Adam/autoencoder/decoder/conv1d_transpose/bias/m4Adam/autoencoder/decoder/conv1d_transpose_1/kernel/m2Adam/autoencoder/decoder/conv1d_transpose_1/bias/m(Adam/autoencoder/encoder/conv1d/kernel/v&Adam/autoencoder/encoder/conv1d/bias/v*Adam/autoencoder/encoder/conv1d_1/kernel/v(Adam/autoencoder/encoder/conv1d_1/bias/v'Adam/autoencoder/encoder/dense/kernel/v%Adam/autoencoder/encoder/dense/bias/v)Adam/autoencoder/encoder/dense_1/kernel/v'Adam/autoencoder/encoder/dense_1/bias/v)Adam/autoencoder/decoder/dense_2/kernel/v'Adam/autoencoder/decoder/dense_2/bias/v2Adam/autoencoder/decoder/conv1d_transpose/kernel/v0Adam/autoencoder/decoder/conv1d_transpose/bias/v4Adam/autoencoder/decoder/conv1d_transpose_1/kernel/v2Adam/autoencoder/decoder/conv1d_transpose_1/bias/v*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_28372??
?}
?
B__inference_decoder_layer_call_and_return_conditional_losses_27943

inputs9
&dense_2_matmul_readvariableop_resource:	?6
'dense_2_biasadd_readvariableop_resource:	?\
Fconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource:>
0conv1d_transpose_biasadd_readvariableop_resource:^
Hconv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource:&@
2conv1d_transpose_1_biasadd_readvariableop_resource:&
identity??'conv1d_transpose/BiasAdd/ReadVariableOp?=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?)conv1d_transpose_1/BiasAdd/ReadVariableOp??conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0z
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????W
reshape/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapedense_2/Relu:activations:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????^
conv1d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv1d_transpose/strided_sliceStridedSliceconv1d_transpose/Shape:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv1d_transpose/strided_slice_1StridedSliceconv1d_transpose/Shape:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
conv1d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv1d_transpose/mulMul)conv1d_transpose/strided_slice_1:output:0conv1d_transpose/mul/y:output:0*
T0*
_output_shapes
: X
conv1d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B : y
conv1d_transpose/addAddV2conv1d_transpose/mul:z:0conv1d_transpose/add/y:output:0*
T0*
_output_shapes
: Z
conv1d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :?
conv1d_transpose/stackPack'conv1d_transpose/strided_slice:output:0conv1d_transpose/add:z:0!conv1d_transpose/stack/2:output:0*
N*
T0*
_output_shapes
:r
0conv1d_transpose/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
,conv1d_transpose/conv1d_transpose/ExpandDims
ExpandDimsreshape/Reshape:output:09conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpFconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0t
2conv1d_transpose/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
.conv1d_transpose/conv1d_transpose/ExpandDims_1
ExpandDimsEconv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0;conv1d_transpose/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
5conv1d_transpose/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7conv1d_transpose/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7conv1d_transpose/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/conv1d_transpose/conv1d_transpose/strided_sliceStridedSliceconv1d_transpose/stack:output:0>conv1d_transpose/conv1d_transpose/strided_slice/stack:output:0@conv1d_transpose/conv1d_transpose/strided_slice/stack_1:output:0@conv1d_transpose/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
7conv1d_transpose/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1conv1d_transpose/conv1d_transpose/strided_slice_1StridedSliceconv1d_transpose/stack:output:0@conv1d_transpose/conv1d_transpose/strided_slice_1/stack:output:0Bconv1d_transpose/conv1d_transpose/strided_slice_1/stack_1:output:0Bconv1d_transpose/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask{
1conv1d_transpose/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:o
-conv1d_transpose/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(conv1d_transpose/conv1d_transpose/concatConcatV28conv1d_transpose/conv1d_transpose/strided_slice:output:0:conv1d_transpose/conv1d_transpose/concat/values_1:output:0:conv1d_transpose/conv1d_transpose/strided_slice_1:output:06conv1d_transpose/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
!conv1d_transpose/conv1d_transposeConv2DBackpropInput1conv1d_transpose/conv1d_transpose/concat:output:07conv1d_transpose/conv1d_transpose/ExpandDims_1:output:05conv1d_transpose/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????2*
paddingVALID*
strides
?
)conv1d_transpose/conv1d_transpose/SqueezeSqueeze*conv1d_transpose/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????2*
squeeze_dims
?
'conv1d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv1d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1d_transpose/BiasAddBiasAdd2conv1d_transpose/conv1d_transpose/Squeeze:output:0/conv1d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2v
conv1d_transpose/ReluRelu!conv1d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2k
conv1d_transpose_1/ShapeShape#conv1d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv1d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv1d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv1d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv1d_transpose_1/strided_sliceStridedSlice!conv1d_transpose_1/Shape:output:0/conv1d_transpose_1/strided_slice/stack:output:01conv1d_transpose_1/strided_slice/stack_1:output:01conv1d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
(conv1d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*conv1d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv1d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv1d_transpose_1/strided_slice_1StridedSlice!conv1d_transpose_1/Shape:output:01conv1d_transpose_1/strided_slice_1/stack:output:03conv1d_transpose_1/strided_slice_1/stack_1:output:03conv1d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv1d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv1d_transpose_1/mulMul+conv1d_transpose_1/strided_slice_1:output:0!conv1d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: Z
conv1d_transpose_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : 
conv1d_transpose_1/addAddV2conv1d_transpose_1/mul:z:0!conv1d_transpose_1/add/y:output:0*
T0*
_output_shapes
: \
conv1d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :&?
conv1d_transpose_1/stackPack)conv1d_transpose_1/strided_slice:output:0conv1d_transpose_1/add:z:0#conv1d_transpose_1/stack/2:output:0*
N*
T0*
_output_shapes
:t
2conv1d_transpose_1/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
.conv1d_transpose_1/conv1d_transpose/ExpandDims
ExpandDims#conv1d_transpose/Relu:activations:0;conv1d_transpose_1/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2?
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:&*
dtype0v
4conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
0conv1d_transpose_1/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:&?
7conv1d_transpose_1/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1conv1d_transpose_1/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_1/stack:output:0@conv1d_transpose_1/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
9conv1d_transpose_1/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3conv1d_transpose_1/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_1/stack:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask}
3conv1d_transpose_1/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:q
/conv1d_transpose_1/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*conv1d_transpose_1/conv1d_transpose/concatConcatV2:conv1d_transpose_1/conv1d_transpose/strided_slice:output:0<conv1d_transpose_1/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_1/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_1/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
#conv1d_transpose_1/conv1d_transposeConv2DBackpropInput3conv1d_transpose_1/conv1d_transpose/concat:output:09conv1d_transpose_1/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_1/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d&*
paddingVALID*
strides
?
+conv1d_transpose_1/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_1/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????d&*
squeeze_dims
?
)conv1d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:&*
dtype0?
conv1d_transpose_1/BiasAddBiasAdd4conv1d_transpose_1/conv1d_transpose/Squeeze:output:01conv1d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d&z
conv1d_transpose_1/ReluRelu#conv1d_transpose_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d&x
IdentityIdentity%conv1d_transpose_1/Relu:activations:0^NoOp*
T0*+
_output_shapes
:?????????d&?
NoOpNoOp(^conv1d_transpose/BiasAdd/ReadVariableOp>^conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp*^conv1d_transpose_1/BiasAdd/ReadVariableOp@^conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2R
'conv1d_transpose/BiasAdd/ReadVariableOp'conv1d_transpose/BiasAdd/ReadVariableOp2~
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2V
)conv1d_transpose_1/BiasAdd/ReadVariableOp)conv1d_transpose_1/BiasAdd/ReadVariableOp2?
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_conv1d_transpose_layer_call_fn_27952

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv1d_transpose_layer_call_and_return_conditional_losses_27068|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?l
?
__inference__traced_save_28215
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop@
<savev2_autoencoder_encoder_conv1d_kernel_read_readvariableop>
:savev2_autoencoder_encoder_conv1d_bias_read_readvariableopB
>savev2_autoencoder_encoder_conv1d_1_kernel_read_readvariableop@
<savev2_autoencoder_encoder_conv1d_1_bias_read_readvariableop?
;savev2_autoencoder_encoder_dense_kernel_read_readvariableop=
9savev2_autoencoder_encoder_dense_bias_read_readvariableopA
=savev2_autoencoder_encoder_dense_1_kernel_read_readvariableop?
;savev2_autoencoder_encoder_dense_1_bias_read_readvariableopA
=savev2_autoencoder_decoder_dense_2_kernel_read_readvariableop?
;savev2_autoencoder_decoder_dense_2_bias_read_readvariableopJ
Fsavev2_autoencoder_decoder_conv1d_transpose_kernel_read_readvariableopH
Dsavev2_autoencoder_decoder_conv1d_transpose_bias_read_readvariableopL
Hsavev2_autoencoder_decoder_conv1d_transpose_1_kernel_read_readvariableopJ
Fsavev2_autoencoder_decoder_conv1d_transpose_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopG
Csavev2_adam_autoencoder_encoder_conv1d_kernel_m_read_readvariableopE
Asavev2_adam_autoencoder_encoder_conv1d_bias_m_read_readvariableopI
Esavev2_adam_autoencoder_encoder_conv1d_1_kernel_m_read_readvariableopG
Csavev2_adam_autoencoder_encoder_conv1d_1_bias_m_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_kernel_m_read_readvariableopD
@savev2_adam_autoencoder_encoder_dense_bias_m_read_readvariableopH
Dsavev2_adam_autoencoder_encoder_dense_1_kernel_m_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_1_bias_m_read_readvariableopH
Dsavev2_adam_autoencoder_decoder_dense_2_kernel_m_read_readvariableopF
Bsavev2_adam_autoencoder_decoder_dense_2_bias_m_read_readvariableopQ
Msavev2_adam_autoencoder_decoder_conv1d_transpose_kernel_m_read_readvariableopO
Ksavev2_adam_autoencoder_decoder_conv1d_transpose_bias_m_read_readvariableopS
Osavev2_adam_autoencoder_decoder_conv1d_transpose_1_kernel_m_read_readvariableopQ
Msavev2_adam_autoencoder_decoder_conv1d_transpose_1_bias_m_read_readvariableopG
Csavev2_adam_autoencoder_encoder_conv1d_kernel_v_read_readvariableopE
Asavev2_adam_autoencoder_encoder_conv1d_bias_v_read_readvariableopI
Esavev2_adam_autoencoder_encoder_conv1d_1_kernel_v_read_readvariableopG
Csavev2_adam_autoencoder_encoder_conv1d_1_bias_v_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_kernel_v_read_readvariableopD
@savev2_adam_autoencoder_encoder_dense_bias_v_read_readvariableopH
Dsavev2_adam_autoencoder_encoder_dense_1_kernel_v_read_readvariableopF
Bsavev2_adam_autoencoder_encoder_dense_1_bias_v_read_readvariableopH
Dsavev2_adam_autoencoder_decoder_dense_2_kernel_v_read_readvariableopF
Bsavev2_adam_autoencoder_decoder_dense_2_bias_v_read_readvariableopQ
Msavev2_adam_autoencoder_decoder_conv1d_transpose_kernel_v_read_readvariableopO
Ksavev2_adam_autoencoder_decoder_conv1d_transpose_bias_v_read_readvariableopS
Osavev2_adam_autoencoder_decoder_conv1d_transpose_1_kernel_v_read_readvariableopQ
Msavev2_adam_autoencoder_decoder_conv1d_transpose_1_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop<savev2_autoencoder_encoder_conv1d_kernel_read_readvariableop:savev2_autoencoder_encoder_conv1d_bias_read_readvariableop>savev2_autoencoder_encoder_conv1d_1_kernel_read_readvariableop<savev2_autoencoder_encoder_conv1d_1_bias_read_readvariableop;savev2_autoencoder_encoder_dense_kernel_read_readvariableop9savev2_autoencoder_encoder_dense_bias_read_readvariableop=savev2_autoencoder_encoder_dense_1_kernel_read_readvariableop;savev2_autoencoder_encoder_dense_1_bias_read_readvariableop=savev2_autoencoder_decoder_dense_2_kernel_read_readvariableop;savev2_autoencoder_decoder_dense_2_bias_read_readvariableopFsavev2_autoencoder_decoder_conv1d_transpose_kernel_read_readvariableopDsavev2_autoencoder_decoder_conv1d_transpose_bias_read_readvariableopHsavev2_autoencoder_decoder_conv1d_transpose_1_kernel_read_readvariableopFsavev2_autoencoder_decoder_conv1d_transpose_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopCsavev2_adam_autoencoder_encoder_conv1d_kernel_m_read_readvariableopAsavev2_adam_autoencoder_encoder_conv1d_bias_m_read_readvariableopEsavev2_adam_autoencoder_encoder_conv1d_1_kernel_m_read_readvariableopCsavev2_adam_autoencoder_encoder_conv1d_1_bias_m_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_kernel_m_read_readvariableop@savev2_adam_autoencoder_encoder_dense_bias_m_read_readvariableopDsavev2_adam_autoencoder_encoder_dense_1_kernel_m_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_1_bias_m_read_readvariableopDsavev2_adam_autoencoder_decoder_dense_2_kernel_m_read_readvariableopBsavev2_adam_autoencoder_decoder_dense_2_bias_m_read_readvariableopMsavev2_adam_autoencoder_decoder_conv1d_transpose_kernel_m_read_readvariableopKsavev2_adam_autoencoder_decoder_conv1d_transpose_bias_m_read_readvariableopOsavev2_adam_autoencoder_decoder_conv1d_transpose_1_kernel_m_read_readvariableopMsavev2_adam_autoencoder_decoder_conv1d_transpose_1_bias_m_read_readvariableopCsavev2_adam_autoencoder_encoder_conv1d_kernel_v_read_readvariableopAsavev2_adam_autoencoder_encoder_conv1d_bias_v_read_readvariableopEsavev2_adam_autoencoder_encoder_conv1d_1_kernel_v_read_readvariableopCsavev2_adam_autoencoder_encoder_conv1d_1_bias_v_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_kernel_v_read_readvariableop@savev2_adam_autoencoder_encoder_dense_bias_v_read_readvariableopDsavev2_adam_autoencoder_encoder_dense_1_kernel_v_read_readvariableopBsavev2_adam_autoencoder_encoder_dense_1_bias_v_read_readvariableopDsavev2_adam_autoencoder_decoder_dense_2_kernel_v_read_readvariableopBsavev2_adam_autoencoder_decoder_dense_2_bias_v_read_readvariableopMsavev2_adam_autoencoder_decoder_conv1d_transpose_kernel_v_read_readvariableopKsavev2_adam_autoencoder_decoder_conv1d_transpose_bias_v_read_readvariableopOsavev2_adam_autoencoder_decoder_conv1d_transpose_1_kernel_v_read_readvariableopMsavev2_adam_autoencoder_decoder_conv1d_transpose_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :&::::	?::	?::	?:?:::&:&: : :&::::	?::	?::	?:?:::&:&:&::::	?::	?::	?:?:::&:&: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:&: 

_output_shapes
::($
"
_output_shapes
:: 	

_output_shapes
::%
!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:!

_output_shapes	
:?:($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:&: 

_output_shapes
:&:

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:&: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:!

_output_shapes	
:?:( $
"
_output_shapes
:: !

_output_shapes
::("$
"
_output_shapes
:&: #

_output_shapes
:&:($$
"
_output_shapes
:&: %

_output_shapes
::(&$
"
_output_shapes
:: '

_output_shapes
::%(!

_output_shapes
:	?: )

_output_shapes
::%*!

_output_shapes
:	?: +

_output_shapes
::%,!

_output_shapes
:	?:!-

_output_shapes	
:?:(.$
"
_output_shapes
:: /

_output_shapes
::(0$
"
_output_shapes
:&: 1

_output_shapes
:&:2

_output_shapes
: 
??
?%
!__inference__traced_restore_28372
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: J
4assignvariableop_5_autoencoder_encoder_conv1d_kernel:&@
2assignvariableop_6_autoencoder_encoder_conv1d_bias:L
6assignvariableop_7_autoencoder_encoder_conv1d_1_kernel:B
4assignvariableop_8_autoencoder_encoder_conv1d_1_bias:F
3assignvariableop_9_autoencoder_encoder_dense_kernel:	?@
2assignvariableop_10_autoencoder_encoder_dense_bias:I
6assignvariableop_11_autoencoder_encoder_dense_1_kernel:	?B
4assignvariableop_12_autoencoder_encoder_dense_1_bias:I
6assignvariableop_13_autoencoder_decoder_dense_2_kernel:	?C
4assignvariableop_14_autoencoder_decoder_dense_2_bias:	?U
?assignvariableop_15_autoencoder_decoder_conv1d_transpose_kernel:K
=assignvariableop_16_autoencoder_decoder_conv1d_transpose_bias:W
Aassignvariableop_17_autoencoder_decoder_conv1d_transpose_1_kernel:&M
?assignvariableop_18_autoencoder_decoder_conv1d_transpose_1_bias:&#
assignvariableop_19_total: #
assignvariableop_20_count: R
<assignvariableop_21_adam_autoencoder_encoder_conv1d_kernel_m:&H
:assignvariableop_22_adam_autoencoder_encoder_conv1d_bias_m:T
>assignvariableop_23_adam_autoencoder_encoder_conv1d_1_kernel_m:J
<assignvariableop_24_adam_autoencoder_encoder_conv1d_1_bias_m:N
;assignvariableop_25_adam_autoencoder_encoder_dense_kernel_m:	?G
9assignvariableop_26_adam_autoencoder_encoder_dense_bias_m:P
=assignvariableop_27_adam_autoencoder_encoder_dense_1_kernel_m:	?I
;assignvariableop_28_adam_autoencoder_encoder_dense_1_bias_m:P
=assignvariableop_29_adam_autoencoder_decoder_dense_2_kernel_m:	?J
;assignvariableop_30_adam_autoencoder_decoder_dense_2_bias_m:	?\
Fassignvariableop_31_adam_autoencoder_decoder_conv1d_transpose_kernel_m:R
Dassignvariableop_32_adam_autoencoder_decoder_conv1d_transpose_bias_m:^
Hassignvariableop_33_adam_autoencoder_decoder_conv1d_transpose_1_kernel_m:&T
Fassignvariableop_34_adam_autoencoder_decoder_conv1d_transpose_1_bias_m:&R
<assignvariableop_35_adam_autoencoder_encoder_conv1d_kernel_v:&H
:assignvariableop_36_adam_autoencoder_encoder_conv1d_bias_v:T
>assignvariableop_37_adam_autoencoder_encoder_conv1d_1_kernel_v:J
<assignvariableop_38_adam_autoencoder_encoder_conv1d_1_bias_v:N
;assignvariableop_39_adam_autoencoder_encoder_dense_kernel_v:	?G
9assignvariableop_40_adam_autoencoder_encoder_dense_bias_v:P
=assignvariableop_41_adam_autoencoder_encoder_dense_1_kernel_v:	?I
;assignvariableop_42_adam_autoencoder_encoder_dense_1_bias_v:P
=assignvariableop_43_adam_autoencoder_decoder_dense_2_kernel_v:	?J
;assignvariableop_44_adam_autoencoder_decoder_dense_2_bias_v:	?\
Fassignvariableop_45_adam_autoencoder_decoder_conv1d_transpose_kernel_v:R
Dassignvariableop_46_adam_autoencoder_decoder_conv1d_transpose_bias_v:^
Hassignvariableop_47_adam_autoencoder_decoder_conv1d_transpose_1_kernel_v:&T
Fassignvariableop_48_adam_autoencoder_decoder_conv1d_transpose_1_bias_v:&
identity_50??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp4assignvariableop_5_autoencoder_encoder_conv1d_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp2assignvariableop_6_autoencoder_encoder_conv1d_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp6assignvariableop_7_autoencoder_encoder_conv1d_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp4assignvariableop_8_autoencoder_encoder_conv1d_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp3assignvariableop_9_autoencoder_encoder_dense_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp2assignvariableop_10_autoencoder_encoder_dense_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp6assignvariableop_11_autoencoder_encoder_dense_1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp4assignvariableop_12_autoencoder_encoder_dense_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp6assignvariableop_13_autoencoder_decoder_dense_2_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp4assignvariableop_14_autoencoder_decoder_dense_2_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp?assignvariableop_15_autoencoder_decoder_conv1d_transpose_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp=assignvariableop_16_autoencoder_decoder_conv1d_transpose_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpAassignvariableop_17_autoencoder_decoder_conv1d_transpose_1_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp?assignvariableop_18_autoencoder_decoder_conv1d_transpose_1_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp<assignvariableop_21_adam_autoencoder_encoder_conv1d_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp:assignvariableop_22_adam_autoencoder_encoder_conv1d_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_autoencoder_encoder_conv1d_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp<assignvariableop_24_adam_autoencoder_encoder_conv1d_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp;assignvariableop_25_adam_autoencoder_encoder_dense_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp9assignvariableop_26_adam_autoencoder_encoder_dense_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp=assignvariableop_27_adam_autoencoder_encoder_dense_1_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp;assignvariableop_28_adam_autoencoder_encoder_dense_1_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp=assignvariableop_29_adam_autoencoder_decoder_dense_2_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp;assignvariableop_30_adam_autoencoder_decoder_dense_2_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpFassignvariableop_31_adam_autoencoder_decoder_conv1d_transpose_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpDassignvariableop_32_adam_autoencoder_decoder_conv1d_transpose_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpHassignvariableop_33_adam_autoencoder_decoder_conv1d_transpose_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOpFassignvariableop_34_adam_autoencoder_decoder_conv1d_transpose_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp<assignvariableop_35_adam_autoencoder_encoder_conv1d_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp:assignvariableop_36_adam_autoencoder_encoder_conv1d_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp>assignvariableop_37_adam_autoencoder_encoder_conv1d_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp<assignvariableop_38_adam_autoencoder_encoder_conv1d_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp;assignvariableop_39_adam_autoencoder_encoder_dense_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp9assignvariableop_40_adam_autoencoder_encoder_dense_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp=assignvariableop_41_adam_autoencoder_encoder_dense_1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp;assignvariableop_42_adam_autoencoder_encoder_dense_1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp=assignvariableop_43_adam_autoencoder_decoder_dense_2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp;assignvariableop_44_adam_autoencoder_decoder_dense_2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOpFassignvariableop_45_adam_autoencoder_decoder_conv1d_transpose_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOpDassignvariableop_46_adam_autoencoder_decoder_conv1d_transpose_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOpHassignvariableop_47_adam_autoencoder_decoder_conv1d_transpose_1_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOpFassignvariableop_48_adam_autoencoder_decoder_conv1d_transpose_1_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_50IdentityIdentity_49:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_50Identity_50:output:0*w
_input_shapesf
d: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_27342

inputs#
encoder_27201:&
encoder_27203:#
encoder_27205:
encoder_27207: 
encoder_27209:	?
encoder_27211: 
encoder_27213:	?
encoder_27215: 
decoder_27317:	?
decoder_27319:	?#
decoder_27321:
decoder_27323:#
decoder_27325:&
decoder_27327:&
identity

identity_1??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_27201encoder_27203encoder_27205encoder_27207encoder_27209encoder_27211encoder_27213encoder_27215*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_27200?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:2decoder_27317decoder_27319decoder_27321decoder_27323decoder_27325decoder_27327*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d&*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_27316l
SquareSquare(encoder/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????r
subSub(encoder/StatefulPartitionedCall:output:1
Square:y:0*
T0*'
_output_shapes
:?????????f
ExpExp(encoder/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:?????????P
sub_1Subsub:z:0Exp:y:0*
T0*'
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Y
addAddV2	sub_1:z:0add/y:output:0*
T0*'
_output_shapes
:?????????V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
MeanMeanadd:z:0Const:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?J
mulMulmul/x:output:0Mean:output:0*
T0*
_output_shapes
: {
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d&G

Identity_1Identitymul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????d&: : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:S O
+
_output_shapes
:?????????d&
 
_user_specified_nameinputs
?
?
+__inference_autoencoder_layer_call_fn_27374
input_1
unknown:&
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
	unknown_7:	?
	unknown_8:	?
	unknown_9:

unknown_10: 

unknown_11:&

unknown_12:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????d&: *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_27342s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????d&: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????d&
!
_user_specified_name	input_1
?
?
'__inference_encoder_layer_call_fn_27765

inputs
unknown:&
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_27200o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d&: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????d&
 
_user_specified_nameinputs
?
?
'__inference_decoder_layer_call_fn_27847

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
	unknown_2:
	unknown_3:&
	unknown_4:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d&*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_27316s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?,
?
M__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_27121

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:&-
biasadd_readvariableop_resource:&
identity??BiasAdd/ReadVariableOp?,conv1d_transpose/ExpandDims_1/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
stack/2Const*
_output_shapes
: *
dtype0*
value	B :&n
stackPackstrided_slice:output:0add:z:0stack/2:output:0*
N*
T0*
_output_shapes
:a
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"???????????????????
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:&*
dtype0c
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:&n
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskj
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:^
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????&*
paddingVALID*
strides
?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????&*
squeeze_dims
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:&*
dtype0?
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????&]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????&n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????&?
NoOpNoOp^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_27540
input_1
unknown:&
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
	unknown_7:	?
	unknown_8:	?
	unknown_9:

unknown_10: 

unknown_11:&

unknown_12:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d&*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_27022s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????d&: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????d&
!
_user_specified_name	input_1
?,
?
K__inference_conv1d_transpose_layer_call_and_return_conditional_losses_27994

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?,conv1d_transpose/ExpandDims_1/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
stack/2Const*
_output_shapes
: *
dtype0*
value	B :n
stackPackstrided_slice:output:0add:z:0stack/2:output:0*
N*
T0*
_output_shapes
:a
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"???????????????????
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0c
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskj
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:^
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_27499
input_1#
encoder_27455:&
encoder_27457:#
encoder_27459:
encoder_27461: 
encoder_27463:	?
encoder_27465: 
encoder_27467:	?
encoder_27469: 
decoder_27474:	?
decoder_27476:	?#
decoder_27478:
decoder_27480:#
decoder_27482:&
decoder_27484:&
identity

identity_1??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_27455encoder_27457encoder_27459encoder_27461encoder_27463encoder_27465encoder_27467encoder_27469*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_27200?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:2decoder_27474decoder_27476decoder_27478decoder_27480decoder_27482decoder_27484*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d&*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_27316l
SquareSquare(encoder/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????r
subSub(encoder/StatefulPartitionedCall:output:1
Square:y:0*
T0*'
_output_shapes
:?????????f
ExpExp(encoder/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:?????????P
sub_1Subsub:z:0Exp:y:0*
T0*'
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Y
addAddV2	sub_1:z:0add/y:output:0*
T0*'
_output_shapes
:?????????V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
MeanMeanadd:z:0Const:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?J
mulMulmul/x:output:0Mean:output:0*
T0*
_output_shapes
: {
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d&G

Identity_1Identitymul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????d&: : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:T P
+
_output_shapes
:?????????d&
!
_user_specified_name	input_1
?,
?
K__inference_conv1d_transpose_layer_call_and_return_conditional_losses_27068

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?,conv1d_transpose/ExpandDims_1/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
stack/2Const*
_output_shapes
: *
dtype0*
value	B :n
stackPackstrided_slice:output:0add:z:0stack/2:output:0*
N*
T0*
_output_shapes
:a
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"???????????????????
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0c
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:n
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskj
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:^
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????*
paddingVALID*
strides
?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????*
squeeze_dims
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :???????????????????
NoOpNoOp^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
2__inference_conv1d_transpose_1_layer_call_fn_28003

inputs
unknown:&
	unknown_0:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_27121|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?,
?
M__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_28045

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:&-
biasadd_readvariableop_resource:&
identity??BiasAdd/ReadVariableOp?,conv1d_transpose/ExpandDims_1/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B : F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
stack/2Const*
_output_shapes
: *
dtype0*
value	B :&n
stackPackstrided_slice:output:0add:z:0stack/2:output:0*
N*
T0*
_output_shapes
:a
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"???????????????????
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:&*
dtype0c
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:&n
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskj
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:^
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"??????????????????&*
paddingVALID*
strides
?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????&*
squeeze_dims
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:&*
dtype0?
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????&]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????&n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????&?
NoOpNoOp^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
+__inference_autoencoder_layer_call_fn_27574

inputs
unknown:&
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
	unknown_7:	?
	unknown_8:	?
	unknown_9:

unknown_10: 

unknown_11:&

unknown_12:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????d&: *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_27342s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????d&: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????d&
 
_user_specified_nameinputs
??
?
F__inference_autoencoder_layer_call_and_return_conditional_losses_27740

inputsP
:encoder_conv1d_conv1d_expanddims_1_readvariableop_resource:&<
.encoder_conv1d_biasadd_readvariableop_resource:R
<encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource:>
0encoder_conv1d_1_biasadd_readvariableop_resource:?
,encoder_dense_matmul_readvariableop_resource:	?;
-encoder_dense_biasadd_readvariableop_resource:A
.encoder_dense_1_matmul_readvariableop_resource:	?=
/encoder_dense_1_biasadd_readvariableop_resource:A
.decoder_dense_2_matmul_readvariableop_resource:	?>
/decoder_dense_2_biasadd_readvariableop_resource:	?d
Ndecoder_conv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource:F
8decoder_conv1d_transpose_biasadd_readvariableop_resource:f
Pdecoder_conv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource:&H
:decoder_conv1d_transpose_1_biasadd_readvariableop_resource:&
identity

identity_1??/decoder/conv1d_transpose/BiasAdd/ReadVariableOp?Edecoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?1decoder/conv1d_transpose_1/BiasAdd/ReadVariableOp?Gdecoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp?&decoder/dense_2/BiasAdd/ReadVariableOp?%decoder/dense_2/MatMul/ReadVariableOp?%encoder/conv1d/BiasAdd/ReadVariableOp?1encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp?'encoder/conv1d_1/BiasAdd/ReadVariableOp?3encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp?$encoder/dense/BiasAdd/ReadVariableOp?#encoder/dense/MatMul/ReadVariableOp?&encoder/dense_1/BiasAdd/ReadVariableOp?%encoder/dense_1/MatMul/ReadVariableOpo
$encoder/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 encoder/conv1d/Conv1D/ExpandDims
ExpandDimsinputs-encoder/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d&?
1encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:encoder_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:&*
dtype0h
&encoder/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"encoder/conv1d/Conv1D/ExpandDims_1
ExpandDims9encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0/encoder/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:&?
encoder/conv1d/Conv1DConv2D)encoder/conv1d/Conv1D/ExpandDims:output:0+encoder/conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????2*
paddingVALID*
strides
?
encoder/conv1d/Conv1D/SqueezeSqueezeencoder/conv1d/Conv1D:output:0*
T0*+
_output_shapes
:?????????2*
squeeze_dims

??????????
%encoder/conv1d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
encoder/conv1d/BiasAddBiasAdd&encoder/conv1d/Conv1D/Squeeze:output:0-encoder/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2r
encoder/conv1d/ReluReluencoder/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2q
&encoder/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"encoder/conv1d_1/Conv1D/ExpandDims
ExpandDims!encoder/conv1d/Relu:activations:0/encoder/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2?
3encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp<encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0j
(encoder/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
$encoder/conv1d_1/Conv1D/ExpandDims_1
ExpandDims;encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:01encoder/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
encoder/conv1d_1/Conv1DConv2D+encoder/conv1d_1/Conv1D/ExpandDims:output:0-encoder/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
encoder/conv1d_1/Conv1D/SqueezeSqueeze encoder/conv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
'encoder/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
encoder/conv1d_1/BiasAddBiasAdd(encoder/conv1d_1/Conv1D/Squeeze:output:0/encoder/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????v
encoder/conv1d_1/ReluRelu!encoder/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????f
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  ?
encoder/flatten/ReshapeReshape#encoder/conv1d_1/Relu:activations:0encoder/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
#encoder/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
encoder/dense/MatMulMatMul encoder/flatten/Reshape:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
%encoder/dense_1/MatMul/ReadVariableOpReadVariableOp.encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
encoder/dense_1/MatMulMatMul encoder/flatten/Reshape:output:0-encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
encoder/dense_1/BiasAddBiasAdd encoder/dense_1/MatMul:product:0.encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
encoder/sampling/ShapeShapeencoder/dense/BiasAdd:output:0*
T0*
_output_shapes
:n
$encoder/sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&encoder/sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&encoder/sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
encoder/sampling/strided_sliceStridedSliceencoder/sampling/Shape:output:0-encoder/sampling/strided_slice/stack:output:0/encoder/sampling/strided_slice/stack_1:output:0/encoder/sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
encoder/sampling/Shape_1Shapeencoder/dense/BiasAdd:output:0*
T0*
_output_shapes
:p
&encoder/sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(encoder/sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(encoder/sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 encoder/sampling/strided_slice_1StridedSlice!encoder/sampling/Shape_1:output:0/encoder/sampling/strided_slice_1/stack:output:01encoder/sampling/strided_slice_1/stack_1:output:01encoder/sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
$encoder/sampling/random_normal/shapePack'encoder/sampling/strided_slice:output:0)encoder/sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:h
#encoder/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%encoder/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
3encoder/sampling/random_normal/RandomStandardNormalRandomStandardNormal-encoder/sampling/random_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2????
"encoder/sampling/random_normal/mulMul<encoder/sampling/random_normal/RandomStandardNormal:output:0.encoder/sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:??????????
encoder/sampling/random_normalAddV2&encoder/sampling/random_normal/mul:z:0,encoder/sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????[
encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
encoder/sampling/mulMulencoder/sampling/mul/x:output:0 encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????g
encoder/sampling/ExpExpencoder/sampling/mul:z:0*
T0*'
_output_shapes
:??????????
encoder/sampling/mul_1Mulencoder/sampling/Exp:y:0"encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:??????????
encoder/sampling/addAddV2encoder/dense/BiasAdd:output:0encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:??????????
%decoder/dense_2/MatMul/ReadVariableOpReadVariableOp.decoder_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
decoder/dense_2/MatMulMatMulencoder/sampling/add:z:0-decoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&decoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
decoder/dense_2/BiasAddBiasAdd decoder/dense_2/MatMul:product:0.decoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
decoder/dense_2/ReluRelu decoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????g
decoder/reshape/ShapeShape"decoder/dense_2/Relu:activations:0*
T0*
_output_shapes
:m
#decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
decoder/reshape/strided_sliceStridedSlicedecoder/reshape/Shape:output:0,decoder/reshape/strided_slice/stack:output:0.decoder/reshape/strided_slice/stack_1:output:0.decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :a
decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
decoder/reshape/Reshape/shapePack&decoder/reshape/strided_slice:output:0(decoder/reshape/Reshape/shape/1:output:0(decoder/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
decoder/reshape/ReshapeReshape"decoder/dense_2/Relu:activations:0&decoder/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????n
decoder/conv1d_transpose/ShapeShape decoder/reshape/Reshape:output:0*
T0*
_output_shapes
:v
,decoder/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.decoder/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.decoder/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&decoder/conv1d_transpose/strided_sliceStridedSlice'decoder/conv1d_transpose/Shape:output:05decoder/conv1d_transpose/strided_slice/stack:output:07decoder/conv1d_transpose/strided_slice/stack_1:output:07decoder/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
.decoder/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv1d_transpose/strided_slice_1StridedSlice'decoder/conv1d_transpose/Shape:output:07decoder/conv1d_transpose/strided_slice_1/stack:output:09decoder/conv1d_transpose/strided_slice_1/stack_1:output:09decoder/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
decoder/conv1d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
decoder/conv1d_transpose/mulMul1decoder/conv1d_transpose/strided_slice_1:output:0'decoder/conv1d_transpose/mul/y:output:0*
T0*
_output_shapes
: `
decoder/conv1d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
decoder/conv1d_transpose/addAddV2 decoder/conv1d_transpose/mul:z:0'decoder/conv1d_transpose/add/y:output:0*
T0*
_output_shapes
: b
 decoder/conv1d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :?
decoder/conv1d_transpose/stackPack/decoder/conv1d_transpose/strided_slice:output:0 decoder/conv1d_transpose/add:z:0)decoder/conv1d_transpose/stack/2:output:0*
N*
T0*
_output_shapes
:z
8decoder/conv1d_transpose/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
4decoder/conv1d_transpose/conv1d_transpose/ExpandDims
ExpandDims decoder/reshape/Reshape:output:0Adecoder/conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
Edecoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpNdecoder_conv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0|
:decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
6decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1
ExpandDimsMdecoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Cdecoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
=decoder/conv1d_transpose/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7decoder/conv1d_transpose/conv1d_transpose/strided_sliceStridedSlice'decoder/conv1d_transpose/stack:output:0Fdecoder/conv1d_transpose/conv1d_transpose/strided_slice/stack:output:0Hdecoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_1:output:0Hdecoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
?decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Adecoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Adecoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9decoder/conv1d_transpose/conv1d_transpose/strided_slice_1StridedSlice'decoder/conv1d_transpose/stack:output:0Hdecoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack:output:0Jdecoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1:output:0Jdecoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
9decoder/conv1d_transpose/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:w
5decoder/conv1d_transpose/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0decoder/conv1d_transpose/conv1d_transpose/concatConcatV2@decoder/conv1d_transpose/conv1d_transpose/strided_slice:output:0Bdecoder/conv1d_transpose/conv1d_transpose/concat/values_1:output:0Bdecoder/conv1d_transpose/conv1d_transpose/strided_slice_1:output:0>decoder/conv1d_transpose/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
)decoder/conv1d_transpose/conv1d_transposeConv2DBackpropInput9decoder/conv1d_transpose/conv1d_transpose/concat:output:0?decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1:output:0=decoder/conv1d_transpose/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????2*
paddingVALID*
strides
?
1decoder/conv1d_transpose/conv1d_transpose/SqueezeSqueeze2decoder/conv1d_transpose/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????2*
squeeze_dims
?
/decoder/conv1d_transpose/BiasAdd/ReadVariableOpReadVariableOp8decoder_conv1d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 decoder/conv1d_transpose/BiasAddBiasAdd:decoder/conv1d_transpose/conv1d_transpose/Squeeze:output:07decoder/conv1d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2?
decoder/conv1d_transpose/ReluRelu)decoder/conv1d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2{
 decoder/conv1d_transpose_1/ShapeShape+decoder/conv1d_transpose/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv1d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv1d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv1d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv1d_transpose_1/strided_sliceStridedSlice)decoder/conv1d_transpose_1/Shape:output:07decoder/conv1d_transpose_1/strided_slice/stack:output:09decoder/conv1d_transpose_1/strided_slice/stack_1:output:09decoder/conv1d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
0decoder/conv1d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv1d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv1d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/conv1d_transpose_1/strided_slice_1StridedSlice)decoder/conv1d_transpose_1/Shape:output:09decoder/conv1d_transpose_1/strided_slice_1/stack:output:0;decoder/conv1d_transpose_1/strided_slice_1/stack_1:output:0;decoder/conv1d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 decoder/conv1d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
decoder/conv1d_transpose_1/mulMul3decoder/conv1d_transpose_1/strided_slice_1:output:0)decoder/conv1d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: b
 decoder/conv1d_transpose_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
decoder/conv1d_transpose_1/addAddV2"decoder/conv1d_transpose_1/mul:z:0)decoder/conv1d_transpose_1/add/y:output:0*
T0*
_output_shapes
: d
"decoder/conv1d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :&?
 decoder/conv1d_transpose_1/stackPack1decoder/conv1d_transpose_1/strided_slice:output:0"decoder/conv1d_transpose_1/add:z:0+decoder/conv1d_transpose_1/stack/2:output:0*
N*
T0*
_output_shapes
:|
:decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
6decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims
ExpandDims+decoder/conv1d_transpose/Relu:activations:0Cdecoder/conv1d_transpose_1/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2?
Gdecoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpPdecoder_conv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:&*
dtype0~
<decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
8decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1
ExpandDimsOdecoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Edecoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:&?
?decoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Adecoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Adecoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9decoder/conv1d_transpose_1/conv1d_transpose/strided_sliceStridedSlice)decoder/conv1d_transpose_1/stack:output:0Hdecoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack:output:0Jdecoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1:output:0Jdecoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Adecoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cdecoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Cdecoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1StridedSlice)decoder/conv1d_transpose_1/stack:output:0Jdecoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack:output:0Ldecoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1:output:0Ldecoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
;decoder/conv1d_transpose_1/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:y
7decoder/conv1d_transpose_1/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2decoder/conv1d_transpose_1/conv1d_transpose/concatConcatV2Bdecoder/conv1d_transpose_1/conv1d_transpose/strided_slice:output:0Ddecoder/conv1d_transpose_1/conv1d_transpose/concat/values_1:output:0Ddecoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1:output:0@decoder/conv1d_transpose_1/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
+decoder/conv1d_transpose_1/conv1d_transposeConv2DBackpropInput;decoder/conv1d_transpose_1/conv1d_transpose/concat:output:0Adecoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1:output:0?decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d&*
paddingVALID*
strides
?
3decoder/conv1d_transpose_1/conv1d_transpose/SqueezeSqueeze4decoder/conv1d_transpose_1/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????d&*
squeeze_dims
?
1decoder/conv1d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv1d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:&*
dtype0?
"decoder/conv1d_transpose_1/BiasAddBiasAdd<decoder/conv1d_transpose_1/conv1d_transpose/Squeeze:output:09decoder/conv1d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d&?
decoder/conv1d_transpose_1/ReluRelu+decoder/conv1d_transpose_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d&b
SquareSquareencoder/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????j
subSub encoder/dense_1/BiasAdd:output:0
Square:y:0*
T0*'
_output_shapes
:?????????^
ExpExp encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????P
sub_1Subsub:z:0Exp:y:0*
T0*'
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Y
addAddV2	sub_1:z:0add/y:output:0*
T0*'
_output_shapes
:?????????V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
MeanMeanadd:z:0Const:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?J
mulMulmul/x:output:0Mean:output:0*
T0*
_output_shapes
: ?
IdentityIdentity-decoder/conv1d_transpose_1/Relu:activations:0^NoOp*
T0*+
_output_shapes
:?????????d&G

Identity_1Identitymul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp0^decoder/conv1d_transpose/BiasAdd/ReadVariableOpF^decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2^decoder/conv1d_transpose_1/BiasAdd/ReadVariableOpH^decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp'^decoder/dense_2/BiasAdd/ReadVariableOp&^decoder/dense_2/MatMul/ReadVariableOp&^encoder/conv1d/BiasAdd/ReadVariableOp2^encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp(^encoder/conv1d_1/BiasAdd/ReadVariableOp4^encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp'^encoder/dense_1/BiasAdd/ReadVariableOp&^encoder/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????d&: : : : : : : : : : : : : : 2b
/decoder/conv1d_transpose/BiasAdd/ReadVariableOp/decoder/conv1d_transpose/BiasAdd/ReadVariableOp2?
Edecoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpEdecoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2f
1decoder/conv1d_transpose_1/BiasAdd/ReadVariableOp1decoder/conv1d_transpose_1/BiasAdd/ReadVariableOp2?
Gdecoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpGdecoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp2P
&decoder/dense_2/BiasAdd/ReadVariableOp&decoder/dense_2/BiasAdd/ReadVariableOp2N
%decoder/dense_2/MatMul/ReadVariableOp%decoder/dense_2/MatMul/ReadVariableOp2N
%encoder/conv1d/BiasAdd/ReadVariableOp%encoder/conv1d/BiasAdd/ReadVariableOp2f
1encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp1encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2R
'encoder/conv1d_1/BiasAdd/ReadVariableOp'encoder/conv1d_1/BiasAdd/ReadVariableOp2j
3encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp3encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2L
$encoder/dense/BiasAdd/ReadVariableOp$encoder/dense/BiasAdd/ReadVariableOp2J
#encoder/dense/MatMul/ReadVariableOp#encoder/dense/MatMul/ReadVariableOp2P
&encoder/dense_1/BiasAdd/ReadVariableOp&encoder/dense_1/BiasAdd/ReadVariableOp2N
%encoder/dense_1/MatMul/ReadVariableOp%encoder/dense_1/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????d&
 
_user_specified_nameinputs
?I
?
B__inference_encoder_layer_call_and_return_conditional_losses_27830

inputsH
2conv1d_conv1d_expanddims_1_readvariableop_resource:&4
&conv1d_biasadd_readvariableop_resource:J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_1_biasadd_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	?3
%dense_biasadd_readvariableop_resource:9
&dense_1_matmul_readvariableop_resource:	?5
'dense_1_biasadd_readvariableop_resource:
identity

identity_1

identity_2??conv1d/BiasAdd/ReadVariableOp?)conv1d/Conv1D/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOpg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d/Conv1D/ExpandDims
ExpandDimsinputs%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d&?
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:&*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:&?
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????2*
paddingVALID*
strides
?
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:?????????2*
squeeze_dims

??????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2b
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d_1/Conv1D/ExpandDims
ExpandDimsconv1d/Relu:activations:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2?
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????f
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  ?
flatten/ReshapeReshapeconv1d_1/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T
sampling/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:f
sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sampling/strided_sliceStridedSlicesampling/Shape:output:0%sampling/strided_slice/stack:output:0'sampling/strided_slice/stack_1:output:0'sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
sampling/Shape_1Shapedense/BiasAdd:output:0*
T0*
_output_shapes
:h
sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:j
 sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sampling/strided_slice_1StridedSlicesampling/Shape_1:output:0'sampling/strided_slice_1/stack:output:0)sampling/strided_slice_1/stack_1:output:0)sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
sampling/random_normal/shapePacksampling/strided_slice:output:0!sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:`
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    b
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
+sampling/random_normal/RandomStandardNormalRandomStandardNormal%sampling/random_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2ַf?
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:??????????
sampling/random_normalAddV2sampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????S
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?x
sampling/mulMulsampling/mul/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????W
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:?????????u
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????s
sampling/addAddV2dense/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????i

Identity_1Identitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????a

Identity_2Identitysampling/add:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d&: : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????d&
 
_user_specified_nameinputs
?I
?
B__inference_encoder_layer_call_and_return_conditional_losses_27200

inputsH
2conv1d_conv1d_expanddims_1_readvariableop_resource:&4
&conv1d_biasadd_readvariableop_resource:J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_1_biasadd_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	?3
%dense_biasadd_readvariableop_resource:9
&dense_1_matmul_readvariableop_resource:	?5
'dense_1_biasadd_readvariableop_resource:
identity

identity_1

identity_2??conv1d/BiasAdd/ReadVariableOp?)conv1d/Conv1D/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOpg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d/Conv1D/ExpandDims
ExpandDimsinputs%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d&?
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:&*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:&?
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????2*
paddingVALID*
strides
?
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:?????????2*
squeeze_dims

??????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2b
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d_1/Conv1D/ExpandDims
ExpandDimsconv1d/Relu:activations:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2?
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????f
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  ?
flatten/ReshapeReshapeconv1d_1/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T
sampling/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:f
sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sampling/strided_sliceStridedSlicesampling/Shape:output:0%sampling/strided_slice/stack:output:0'sampling/strided_slice/stack_1:output:0'sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
sampling/Shape_1Shapedense/BiasAdd:output:0*
T0*
_output_shapes
:h
sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:j
 sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sampling/strided_slice_1StridedSlicesampling/Shape_1:output:0'sampling/strided_slice_1/stack:output:0)sampling/strided_slice_1/stack_1:output:0)sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
sampling/random_normal/shapePacksampling/strided_slice:output:0!sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:`
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    b
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
+sampling/random_normal/RandomStandardNormalRandomStandardNormal%sampling/random_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2н??
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:??????????
sampling/random_normalAddV2sampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????S
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?x
sampling/mulMulsampling/mul/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????W
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:?????????u
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????s
sampling/addAddV2dense/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????i

Identity_1Identitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????a

Identity_2Identitysampling/add:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d&: : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????d&
 
_user_specified_nameinputs
ڃ
?
 __inference__wrapped_model_27022
input_1\
Fautoencoder_encoder_conv1d_conv1d_expanddims_1_readvariableop_resource:&H
:autoencoder_encoder_conv1d_biasadd_readvariableop_resource:^
Hautoencoder_encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource:J
<autoencoder_encoder_conv1d_1_biasadd_readvariableop_resource:K
8autoencoder_encoder_dense_matmul_readvariableop_resource:	?G
9autoencoder_encoder_dense_biasadd_readvariableop_resource:M
:autoencoder_encoder_dense_1_matmul_readvariableop_resource:	?I
;autoencoder_encoder_dense_1_biasadd_readvariableop_resource:M
:autoencoder_decoder_dense_2_matmul_readvariableop_resource:	?J
;autoencoder_decoder_dense_2_biasadd_readvariableop_resource:	?p
Zautoencoder_decoder_conv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource:R
Dautoencoder_decoder_conv1d_transpose_biasadd_readvariableop_resource:r
\autoencoder_decoder_conv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource:&T
Fautoencoder_decoder_conv1d_transpose_1_biasadd_readvariableop_resource:&
identity??;autoencoder/decoder/conv1d_transpose/BiasAdd/ReadVariableOp?Qautoencoder/decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?=autoencoder/decoder/conv1d_transpose_1/BiasAdd/ReadVariableOp?Sautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp?2autoencoder/decoder/dense_2/BiasAdd/ReadVariableOp?1autoencoder/decoder/dense_2/MatMul/ReadVariableOp?1autoencoder/encoder/conv1d/BiasAdd/ReadVariableOp?=autoencoder/encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp?3autoencoder/encoder/conv1d_1/BiasAdd/ReadVariableOp??autoencoder/encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp?0autoencoder/encoder/dense/BiasAdd/ReadVariableOp?/autoencoder/encoder/dense/MatMul/ReadVariableOp?2autoencoder/encoder/dense_1/BiasAdd/ReadVariableOp?1autoencoder/encoder/dense_1/MatMul/ReadVariableOp{
0autoencoder/encoder/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
,autoencoder/encoder/conv1d/Conv1D/ExpandDims
ExpandDimsinput_19autoencoder/encoder/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d&?
=autoencoder/encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpFautoencoder_encoder_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:&*
dtype0t
2autoencoder/encoder/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
.autoencoder/encoder/conv1d/Conv1D/ExpandDims_1
ExpandDimsEautoencoder/encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0;autoencoder/encoder/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:&?
!autoencoder/encoder/conv1d/Conv1DConv2D5autoencoder/encoder/conv1d/Conv1D/ExpandDims:output:07autoencoder/encoder/conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????2*
paddingVALID*
strides
?
)autoencoder/encoder/conv1d/Conv1D/SqueezeSqueeze*autoencoder/encoder/conv1d/Conv1D:output:0*
T0*+
_output_shapes
:?????????2*
squeeze_dims

??????????
1autoencoder/encoder/conv1d/BiasAdd/ReadVariableOpReadVariableOp:autoencoder_encoder_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
"autoencoder/encoder/conv1d/BiasAddBiasAdd2autoencoder/encoder/conv1d/Conv1D/Squeeze:output:09autoencoder/encoder/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2?
autoencoder/encoder/conv1d/ReluRelu+autoencoder/encoder/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2}
2autoencoder/encoder/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.autoencoder/encoder/conv1d_1/Conv1D/ExpandDims
ExpandDims-autoencoder/encoder/conv1d/Relu:activations:0;autoencoder/encoder/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2?
?autoencoder/encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpHautoencoder_encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0v
4autoencoder/encoder/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
0autoencoder/encoder/conv1d_1/Conv1D/ExpandDims_1
ExpandDimsGautoencoder/encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0=autoencoder/encoder/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
#autoencoder/encoder/conv1d_1/Conv1DConv2D7autoencoder/encoder/conv1d_1/Conv1D/ExpandDims:output:09autoencoder/encoder/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
+autoencoder/encoder/conv1d_1/Conv1D/SqueezeSqueeze,autoencoder/encoder/conv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
3autoencoder/encoder/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_encoder_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$autoencoder/encoder/conv1d_1/BiasAddBiasAdd4autoencoder/encoder/conv1d_1/Conv1D/Squeeze:output:0;autoencoder/encoder/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????
!autoencoder/encoder/conv1d_1/ReluRelu-autoencoder/encoder/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????r
!autoencoder/encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  ?
#autoencoder/encoder/flatten/ReshapeReshape/autoencoder/encoder/conv1d_1/Relu:activations:0*autoencoder/encoder/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
/autoencoder/encoder/dense/MatMul/ReadVariableOpReadVariableOp8autoencoder_encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
 autoencoder/encoder/dense/MatMulMatMul,autoencoder/encoder/flatten/Reshape:output:07autoencoder/encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
0autoencoder/encoder/dense/BiasAdd/ReadVariableOpReadVariableOp9autoencoder_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
!autoencoder/encoder/dense/BiasAddBiasAdd*autoencoder/encoder/dense/MatMul:product:08autoencoder/encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
1autoencoder/encoder/dense_1/MatMul/ReadVariableOpReadVariableOp:autoencoder_encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
"autoencoder/encoder/dense_1/MatMulMatMul,autoencoder/encoder/flatten/Reshape:output:09autoencoder/encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
2autoencoder/encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
#autoencoder/encoder/dense_1/BiasAddBiasAdd,autoencoder/encoder/dense_1/MatMul:product:0:autoencoder/encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
"autoencoder/encoder/sampling/ShapeShape*autoencoder/encoder/dense/BiasAdd:output:0*
T0*
_output_shapes
:z
0autoencoder/encoder/sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2autoencoder/encoder/sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2autoencoder/encoder/sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*autoencoder/encoder/sampling/strided_sliceStridedSlice+autoencoder/encoder/sampling/Shape:output:09autoencoder/encoder/sampling/strided_slice/stack:output:0;autoencoder/encoder/sampling/strided_slice/stack_1:output:0;autoencoder/encoder/sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
$autoencoder/encoder/sampling/Shape_1Shape*autoencoder/encoder/dense/BiasAdd:output:0*
T0*
_output_shapes
:|
2autoencoder/encoder/sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4autoencoder/encoder/sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4autoencoder/encoder/sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,autoencoder/encoder/sampling/strided_slice_1StridedSlice-autoencoder/encoder/sampling/Shape_1:output:0;autoencoder/encoder/sampling/strided_slice_1/stack:output:0=autoencoder/encoder/sampling/strided_slice_1/stack_1:output:0=autoencoder/encoder/sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0autoencoder/encoder/sampling/random_normal/shapePack3autoencoder/encoder/sampling/strided_slice:output:05autoencoder/encoder/sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:t
/autoencoder/encoder/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    v
1autoencoder/encoder/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?autoencoder/encoder/sampling/random_normal/RandomStandardNormalRandomStandardNormal9autoencoder/encoder/sampling/random_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2̀??
.autoencoder/encoder/sampling/random_normal/mulMulHautoencoder/encoder/sampling/random_normal/RandomStandardNormal:output:0:autoencoder/encoder/sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:??????????
*autoencoder/encoder/sampling/random_normalAddV22autoencoder/encoder/sampling/random_normal/mul:z:08autoencoder/encoder/sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????g
"autoencoder/encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
 autoencoder/encoder/sampling/mulMul+autoencoder/encoder/sampling/mul/x:output:0,autoencoder/encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
 autoencoder/encoder/sampling/ExpExp$autoencoder/encoder/sampling/mul:z:0*
T0*'
_output_shapes
:??????????
"autoencoder/encoder/sampling/mul_1Mul$autoencoder/encoder/sampling/Exp:y:0.autoencoder/encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:??????????
 autoencoder/encoder/sampling/addAddV2*autoencoder/encoder/dense/BiasAdd:output:0&autoencoder/encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:??????????
1autoencoder/decoder/dense_2/MatMul/ReadVariableOpReadVariableOp:autoencoder_decoder_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
"autoencoder/decoder/dense_2/MatMulMatMul$autoencoder/encoder/sampling/add:z:09autoencoder/decoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
2autoencoder/decoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_decoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
#autoencoder/decoder/dense_2/BiasAddBiasAdd,autoencoder/decoder/dense_2/MatMul:product:0:autoencoder/decoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
 autoencoder/decoder/dense_2/ReluRelu,autoencoder/decoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????
!autoencoder/decoder/reshape/ShapeShape.autoencoder/decoder/dense_2/Relu:activations:0*
T0*
_output_shapes
:y
/autoencoder/decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1autoencoder/decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1autoencoder/decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)autoencoder/decoder/reshape/strided_sliceStridedSlice*autoencoder/decoder/reshape/Shape:output:08autoencoder/decoder/reshape/strided_slice/stack:output:0:autoencoder/decoder/reshape/strided_slice/stack_1:output:0:autoencoder/decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+autoencoder/decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :m
+autoencoder/decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
)autoencoder/decoder/reshape/Reshape/shapePack2autoencoder/decoder/reshape/strided_slice:output:04autoencoder/decoder/reshape/Reshape/shape/1:output:04autoencoder/decoder/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
#autoencoder/decoder/reshape/ReshapeReshape.autoencoder/decoder/dense_2/Relu:activations:02autoencoder/decoder/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:??????????
*autoencoder/decoder/conv1d_transpose/ShapeShape,autoencoder/decoder/reshape/Reshape:output:0*
T0*
_output_shapes
:?
8autoencoder/decoder/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:autoencoder/decoder/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:autoencoder/decoder/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2autoencoder/decoder/conv1d_transpose/strided_sliceStridedSlice3autoencoder/decoder/conv1d_transpose/Shape:output:0Aautoencoder/decoder/conv1d_transpose/strided_slice/stack:output:0Cautoencoder/decoder/conv1d_transpose/strided_slice/stack_1:output:0Cautoencoder/decoder/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:autoencoder/decoder/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
<autoencoder/decoder/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<autoencoder/decoder/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4autoencoder/decoder/conv1d_transpose/strided_slice_1StridedSlice3autoencoder/decoder/conv1d_transpose/Shape:output:0Cautoencoder/decoder/conv1d_transpose/strided_slice_1/stack:output:0Eautoencoder/decoder/conv1d_transpose/strided_slice_1/stack_1:output:0Eautoencoder/decoder/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*autoencoder/decoder/conv1d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
(autoencoder/decoder/conv1d_transpose/mulMul=autoencoder/decoder/conv1d_transpose/strided_slice_1:output:03autoencoder/decoder/conv1d_transpose/mul/y:output:0*
T0*
_output_shapes
: l
*autoencoder/decoder/conv1d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
(autoencoder/decoder/conv1d_transpose/addAddV2,autoencoder/decoder/conv1d_transpose/mul:z:03autoencoder/decoder/conv1d_transpose/add/y:output:0*
T0*
_output_shapes
: n
,autoencoder/decoder/conv1d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :?
*autoencoder/decoder/conv1d_transpose/stackPack;autoencoder/decoder/conv1d_transpose/strided_slice:output:0,autoencoder/decoder/conv1d_transpose/add:z:05autoencoder/decoder/conv1d_transpose/stack/2:output:0*
N*
T0*
_output_shapes
:?
Dautoencoder/decoder/conv1d_transpose/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
@autoencoder/decoder/conv1d_transpose/conv1d_transpose/ExpandDims
ExpandDims,autoencoder/decoder/reshape/Reshape:output:0Mautoencoder/decoder/conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
Qautoencoder/decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpZautoencoder_decoder_conv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0?
Fautoencoder/decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Bautoencoder/decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1
ExpandDimsYautoencoder/decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Oautoencoder/decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:?
Iautoencoder/decoder/conv1d_transpose/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Kautoencoder/decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Kautoencoder/decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Cautoencoder/decoder/conv1d_transpose/conv1d_transpose/strided_sliceStridedSlice3autoencoder/decoder/conv1d_transpose/stack:output:0Rautoencoder/decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack:output:0Tautoencoder/decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_1:output:0Tautoencoder/decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Kautoencoder/decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Mautoencoder/decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Mautoencoder/decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Eautoencoder/decoder/conv1d_transpose/conv1d_transpose/strided_slice_1StridedSlice3autoencoder/decoder/conv1d_transpose/stack:output:0Tautoencoder/decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack:output:0Vautoencoder/decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1:output:0Vautoencoder/decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Eautoencoder/decoder/conv1d_transpose/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:?
Aautoencoder/decoder/conv1d_transpose/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<autoencoder/decoder/conv1d_transpose/conv1d_transpose/concatConcatV2Lautoencoder/decoder/conv1d_transpose/conv1d_transpose/strided_slice:output:0Nautoencoder/decoder/conv1d_transpose/conv1d_transpose/concat/values_1:output:0Nautoencoder/decoder/conv1d_transpose/conv1d_transpose/strided_slice_1:output:0Jautoencoder/decoder/conv1d_transpose/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5autoencoder/decoder/conv1d_transpose/conv1d_transposeConv2DBackpropInputEautoencoder/decoder/conv1d_transpose/conv1d_transpose/concat:output:0Kautoencoder/decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1:output:0Iautoencoder/decoder/conv1d_transpose/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????2*
paddingVALID*
strides
?
=autoencoder/decoder/conv1d_transpose/conv1d_transpose/SqueezeSqueeze>autoencoder/decoder/conv1d_transpose/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????2*
squeeze_dims
?
;autoencoder/decoder/conv1d_transpose/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_decoder_conv1d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
,autoencoder/decoder/conv1d_transpose/BiasAddBiasAddFautoencoder/decoder/conv1d_transpose/conv1d_transpose/Squeeze:output:0Cautoencoder/decoder/conv1d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2?
)autoencoder/decoder/conv1d_transpose/ReluRelu5autoencoder/decoder/conv1d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2?
,autoencoder/decoder/conv1d_transpose_1/ShapeShape7autoencoder/decoder/conv1d_transpose/Relu:activations:0*
T0*
_output_shapes
:?
:autoencoder/decoder/conv1d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<autoencoder/decoder/conv1d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<autoencoder/decoder/conv1d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4autoencoder/decoder/conv1d_transpose_1/strided_sliceStridedSlice5autoencoder/decoder/conv1d_transpose_1/Shape:output:0Cautoencoder/decoder/conv1d_transpose_1/strided_slice/stack:output:0Eautoencoder/decoder/conv1d_transpose_1/strided_slice/stack_1:output:0Eautoencoder/decoder/conv1d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
<autoencoder/decoder/conv1d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
>autoencoder/decoder/conv1d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>autoencoder/decoder/conv1d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6autoencoder/decoder/conv1d_transpose_1/strided_slice_1StridedSlice5autoencoder/decoder/conv1d_transpose_1/Shape:output:0Eautoencoder/decoder/conv1d_transpose_1/strided_slice_1/stack:output:0Gautoencoder/decoder/conv1d_transpose_1/strided_slice_1/stack_1:output:0Gautoencoder/decoder/conv1d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,autoencoder/decoder/conv1d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
*autoencoder/decoder/conv1d_transpose_1/mulMul?autoencoder/decoder/conv1d_transpose_1/strided_slice_1:output:05autoencoder/decoder/conv1d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: n
,autoencoder/decoder/conv1d_transpose_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
*autoencoder/decoder/conv1d_transpose_1/addAddV2.autoencoder/decoder/conv1d_transpose_1/mul:z:05autoencoder/decoder/conv1d_transpose_1/add/y:output:0*
T0*
_output_shapes
: p
.autoencoder/decoder/conv1d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :&?
,autoencoder/decoder/conv1d_transpose_1/stackPack=autoencoder/decoder/conv1d_transpose_1/strided_slice:output:0.autoencoder/decoder/conv1d_transpose_1/add:z:07autoencoder/decoder/conv1d_transpose_1/stack/2:output:0*
N*
T0*
_output_shapes
:?
Fautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
Bautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims
ExpandDims7autoencoder/decoder/conv1d_transpose/Relu:activations:0Oautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2?
Sautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp\autoencoder_decoder_conv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:&*
dtype0?
Hautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Dautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1
ExpandDims[autoencoder/decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Qautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:&?
Kautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Mautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Mautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Eautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/strided_sliceStridedSlice5autoencoder/decoder/conv1d_transpose_1/stack:output:0Tautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack:output:0Vautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1:output:0Vautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Mautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Oautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Oautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Gautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1StridedSlice5autoencoder/decoder/conv1d_transpose_1/stack:output:0Vautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack:output:0Xautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1:output:0Xautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Gautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>autoencoder/decoder/conv1d_transpose_1/conv1d_transpose/concatConcatV2Nautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/strided_slice:output:0Pautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/concat/values_1:output:0Pautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1:output:0Lautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
7autoencoder/decoder/conv1d_transpose_1/conv1d_transposeConv2DBackpropInputGautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/concat:output:0Mautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1:output:0Kautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d&*
paddingVALID*
strides
?
?autoencoder/decoder/conv1d_transpose_1/conv1d_transpose/SqueezeSqueeze@autoencoder/decoder/conv1d_transpose_1/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????d&*
squeeze_dims
?
=autoencoder/decoder/conv1d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpFautoencoder_decoder_conv1d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:&*
dtype0?
.autoencoder/decoder/conv1d_transpose_1/BiasAddBiasAddHautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/Squeeze:output:0Eautoencoder/decoder/conv1d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d&?
+autoencoder/decoder/conv1d_transpose_1/ReluRelu7autoencoder/decoder/conv1d_transpose_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d&z
autoencoder/SquareSquare*autoencoder/encoder/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
autoencoder/subSub,autoencoder/encoder/dense_1/BiasAdd:output:0autoencoder/Square:y:0*
T0*'
_output_shapes
:?????????v
autoencoder/ExpExp,autoencoder/encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????t
autoencoder/sub_1Subautoencoder/sub:z:0autoencoder/Exp:y:0*
T0*'
_output_shapes
:?????????V
autoencoder/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??}
autoencoder/addAddV2autoencoder/sub_1:z:0autoencoder/add/y:output:0*
T0*'
_output_shapes
:?????????b
autoencoder/ConstConst*
_output_shapes
:*
dtype0*
valueB"       j
autoencoder/MeanMeanautoencoder/add:z:0autoencoder/Const:output:0*
T0*
_output_shapes
: V
autoencoder/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?n
autoencoder/mulMulautoencoder/mul/x:output:0autoencoder/Mean:output:0*
T0*
_output_shapes
: ?
IdentityIdentity9autoencoder/decoder/conv1d_transpose_1/Relu:activations:0^NoOp*
T0*+
_output_shapes
:?????????d&?
NoOpNoOp<^autoencoder/decoder/conv1d_transpose/BiasAdd/ReadVariableOpR^autoencoder/decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp>^autoencoder/decoder/conv1d_transpose_1/BiasAdd/ReadVariableOpT^autoencoder/decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp3^autoencoder/decoder/dense_2/BiasAdd/ReadVariableOp2^autoencoder/decoder/dense_2/MatMul/ReadVariableOp2^autoencoder/encoder/conv1d/BiasAdd/ReadVariableOp>^autoencoder/encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp4^autoencoder/encoder/conv1d_1/BiasAdd/ReadVariableOp@^autoencoder/encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp1^autoencoder/encoder/dense/BiasAdd/ReadVariableOp0^autoencoder/encoder/dense/MatMul/ReadVariableOp3^autoencoder/encoder/dense_1/BiasAdd/ReadVariableOp2^autoencoder/encoder/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????d&: : : : : : : : : : : : : : 2z
;autoencoder/decoder/conv1d_transpose/BiasAdd/ReadVariableOp;autoencoder/decoder/conv1d_transpose/BiasAdd/ReadVariableOp2?
Qautoencoder/decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpQautoencoder/decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2~
=autoencoder/decoder/conv1d_transpose_1/BiasAdd/ReadVariableOp=autoencoder/decoder/conv1d_transpose_1/BiasAdd/ReadVariableOp2?
Sautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpSautoencoder/decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp2h
2autoencoder/decoder/dense_2/BiasAdd/ReadVariableOp2autoencoder/decoder/dense_2/BiasAdd/ReadVariableOp2f
1autoencoder/decoder/dense_2/MatMul/ReadVariableOp1autoencoder/decoder/dense_2/MatMul/ReadVariableOp2f
1autoencoder/encoder/conv1d/BiasAdd/ReadVariableOp1autoencoder/encoder/conv1d/BiasAdd/ReadVariableOp2~
=autoencoder/encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp=autoencoder/encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2j
3autoencoder/encoder/conv1d_1/BiasAdd/ReadVariableOp3autoencoder/encoder/conv1d_1/BiasAdd/ReadVariableOp2?
?autoencoder/encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp?autoencoder/encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2d
0autoencoder/encoder/dense/BiasAdd/ReadVariableOp0autoencoder/encoder/dense/BiasAdd/ReadVariableOp2b
/autoencoder/encoder/dense/MatMul/ReadVariableOp/autoencoder/encoder/dense/MatMul/ReadVariableOp2h
2autoencoder/encoder/dense_1/BiasAdd/ReadVariableOp2autoencoder/encoder/dense_1/BiasAdd/ReadVariableOp2f
1autoencoder/encoder/dense_1/MatMul/ReadVariableOp1autoencoder/encoder/dense_1/MatMul/ReadVariableOp:T P
+
_output_shapes
:?????????d&
!
_user_specified_name	input_1
?}
?
B__inference_decoder_layer_call_and_return_conditional_losses_27316

inputs9
&dense_2_matmul_readvariableop_resource:	?6
'dense_2_biasadd_readvariableop_resource:	?\
Fconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource:>
0conv1d_transpose_biasadd_readvariableop_resource:^
Hconv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource:&@
2conv1d_transpose_1_biasadd_readvariableop_resource:&
identity??'conv1d_transpose/BiasAdd/ReadVariableOp?=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?)conv1d_transpose_1/BiasAdd/ReadVariableOp??conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0z
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????W
reshape/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapedense_2/Relu:activations:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????^
conv1d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv1d_transpose/strided_sliceStridedSliceconv1d_transpose/Shape:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv1d_transpose/strided_slice_1StridedSliceconv1d_transpose/Shape:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
conv1d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv1d_transpose/mulMul)conv1d_transpose/strided_slice_1:output:0conv1d_transpose/mul/y:output:0*
T0*
_output_shapes
: X
conv1d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B : y
conv1d_transpose/addAddV2conv1d_transpose/mul:z:0conv1d_transpose/add/y:output:0*
T0*
_output_shapes
: Z
conv1d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :?
conv1d_transpose/stackPack'conv1d_transpose/strided_slice:output:0conv1d_transpose/add:z:0!conv1d_transpose/stack/2:output:0*
N*
T0*
_output_shapes
:r
0conv1d_transpose/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
,conv1d_transpose/conv1d_transpose/ExpandDims
ExpandDimsreshape/Reshape:output:09conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpFconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0t
2conv1d_transpose/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
.conv1d_transpose/conv1d_transpose/ExpandDims_1
ExpandDimsEconv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0;conv1d_transpose/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
5conv1d_transpose/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7conv1d_transpose/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7conv1d_transpose/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/conv1d_transpose/conv1d_transpose/strided_sliceStridedSliceconv1d_transpose/stack:output:0>conv1d_transpose/conv1d_transpose/strided_slice/stack:output:0@conv1d_transpose/conv1d_transpose/strided_slice/stack_1:output:0@conv1d_transpose/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
7conv1d_transpose/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
9conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1conv1d_transpose/conv1d_transpose/strided_slice_1StridedSliceconv1d_transpose/stack:output:0@conv1d_transpose/conv1d_transpose/strided_slice_1/stack:output:0Bconv1d_transpose/conv1d_transpose/strided_slice_1/stack_1:output:0Bconv1d_transpose/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask{
1conv1d_transpose/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:o
-conv1d_transpose/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
(conv1d_transpose/conv1d_transpose/concatConcatV28conv1d_transpose/conv1d_transpose/strided_slice:output:0:conv1d_transpose/conv1d_transpose/concat/values_1:output:0:conv1d_transpose/conv1d_transpose/strided_slice_1:output:06conv1d_transpose/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
!conv1d_transpose/conv1d_transposeConv2DBackpropInput1conv1d_transpose/conv1d_transpose/concat:output:07conv1d_transpose/conv1d_transpose/ExpandDims_1:output:05conv1d_transpose/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????2*
paddingVALID*
strides
?
)conv1d_transpose/conv1d_transpose/SqueezeSqueeze*conv1d_transpose/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????2*
squeeze_dims
?
'conv1d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv1d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1d_transpose/BiasAddBiasAdd2conv1d_transpose/conv1d_transpose/Squeeze:output:0/conv1d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2v
conv1d_transpose/ReluRelu!conv1d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2k
conv1d_transpose_1/ShapeShape#conv1d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv1d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv1d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv1d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv1d_transpose_1/strided_sliceStridedSlice!conv1d_transpose_1/Shape:output:0/conv1d_transpose_1/strided_slice/stack:output:01conv1d_transpose_1/strided_slice/stack_1:output:01conv1d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
(conv1d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*conv1d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv1d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv1d_transpose_1/strided_slice_1StridedSlice!conv1d_transpose_1/Shape:output:01conv1d_transpose_1/strided_slice_1/stack:output:03conv1d_transpose_1/strided_slice_1/stack_1:output:03conv1d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv1d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv1d_transpose_1/mulMul+conv1d_transpose_1/strided_slice_1:output:0!conv1d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: Z
conv1d_transpose_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : 
conv1d_transpose_1/addAddV2conv1d_transpose_1/mul:z:0!conv1d_transpose_1/add/y:output:0*
T0*
_output_shapes
: \
conv1d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :&?
conv1d_transpose_1/stackPack)conv1d_transpose_1/strided_slice:output:0conv1d_transpose_1/add:z:0#conv1d_transpose_1/stack/2:output:0*
N*
T0*
_output_shapes
:t
2conv1d_transpose_1/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
.conv1d_transpose_1/conv1d_transpose/ExpandDims
ExpandDims#conv1d_transpose/Relu:activations:0;conv1d_transpose_1/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2?
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:&*
dtype0v
4conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
0conv1d_transpose_1/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:&?
7conv1d_transpose_1/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1conv1d_transpose_1/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_1/stack:output:0@conv1d_transpose_1/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
9conv1d_transpose_1/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
;conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3conv1d_transpose_1/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_1/stack:output:0Bconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask}
3conv1d_transpose_1/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:q
/conv1d_transpose_1/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*conv1d_transpose_1/conv1d_transpose/concatConcatV2:conv1d_transpose_1/conv1d_transpose/strided_slice:output:0<conv1d_transpose_1/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_1/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_1/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
#conv1d_transpose_1/conv1d_transposeConv2DBackpropInput3conv1d_transpose_1/conv1d_transpose/concat:output:09conv1d_transpose_1/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_1/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d&*
paddingVALID*
strides
?
+conv1d_transpose_1/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_1/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????d&*
squeeze_dims
?
)conv1d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:&*
dtype0?
conv1d_transpose_1/BiasAddBiasAdd4conv1d_transpose_1/conv1d_transpose/Squeeze:output:01conv1d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d&z
conv1d_transpose_1/ReluRelu#conv1d_transpose_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d&x
IdentityIdentity%conv1d_transpose_1/Relu:activations:0^NoOp*
T0*+
_output_shapes
:?????????d&?
NoOpNoOp(^conv1d_transpose/BiasAdd/ReadVariableOp>^conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp*^conv1d_transpose_1/BiasAdd/ReadVariableOp@^conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2R
'conv1d_transpose/BiasAdd/ReadVariableOp'conv1d_transpose/BiasAdd/ReadVariableOp2~
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2V
)conv1d_transpose_1/BiasAdd/ReadVariableOp)conv1d_transpose_1/BiasAdd/ReadVariableOp2?
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_14
serving_default_input_1:0?????????d&@
output_14
StatefulPartitionedCall:0?????????d&tensorflow/serving/predict:??
?
encoder
decoder
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_model
?
		conv0
	
conv1
flat

dense_mean
dense_log_var
sampling
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	dense
reshape
	conv1
	conv2
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
iter

beta_1

beta_2
	decay
learning_rate m?!m?"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?-m? v?!v?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?-v?"
	optimizer
?
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13"
trackable_list_wrapper
?
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

 kernel
!bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

"kernel
#bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

$kernel
%bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

&kernel
'bias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
X
 0
!1
"2
#3
$4
%5
&6
'7"
trackable_list_wrapper
X
 0
!1
"2
#3
$4
%5
&6
'7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

(kernel
)bias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

*kernel
+bias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

,kernel
-bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
J
(0
)1
*2
+3
,4
-5"
trackable_list_wrapper
J
(0
)1
*2
+3
,4
-5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
7:5&2!autoencoder/encoder/conv1d/kernel
-:+2autoencoder/encoder/conv1d/bias
9:72#autoencoder/encoder/conv1d_1/kernel
/:-2!autoencoder/encoder/conv1d_1/bias
3:1	?2 autoencoder/encoder/dense/kernel
,:*2autoencoder/encoder/dense/bias
5:3	?2"autoencoder/encoder/dense_1/kernel
.:,2 autoencoder/encoder/dense_1/bias
5:3	?2"autoencoder/decoder/dense_2/kernel
/:-?2 autoencoder/decoder/dense_2/bias
A:?2+autoencoder/decoder/conv1d_transpose/kernel
7:52)autoencoder/decoder/conv1d_transpose/bias
C:A&2-autoencoder/decoder/conv1d_transpose_1/kernel
9:7&2+autoencoder/decoder/conv1d_transpose_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
e0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
3	variables
4trainable_variables
5regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
7	variables
8trainable_variables
9regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
?	variables
@trainable_variables
Aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
	0

1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
<::&2(Adam/autoencoder/encoder/conv1d/kernel/m
2:02&Adam/autoencoder/encoder/conv1d/bias/m
>:<2*Adam/autoencoder/encoder/conv1d_1/kernel/m
4:22(Adam/autoencoder/encoder/conv1d_1/bias/m
8:6	?2'Adam/autoencoder/encoder/dense/kernel/m
1:/2%Adam/autoencoder/encoder/dense/bias/m
::8	?2)Adam/autoencoder/encoder/dense_1/kernel/m
3:12'Adam/autoencoder/encoder/dense_1/bias/m
::8	?2)Adam/autoencoder/decoder/dense_2/kernel/m
4:2?2'Adam/autoencoder/decoder/dense_2/bias/m
F:D22Adam/autoencoder/decoder/conv1d_transpose/kernel/m
<::20Adam/autoencoder/decoder/conv1d_transpose/bias/m
H:F&24Adam/autoencoder/decoder/conv1d_transpose_1/kernel/m
>:<&22Adam/autoencoder/decoder/conv1d_transpose_1/bias/m
<::&2(Adam/autoencoder/encoder/conv1d/kernel/v
2:02&Adam/autoencoder/encoder/conv1d/bias/v
>:<2*Adam/autoencoder/encoder/conv1d_1/kernel/v
4:22(Adam/autoencoder/encoder/conv1d_1/bias/v
8:6	?2'Adam/autoencoder/encoder/dense/kernel/v
1:/2%Adam/autoencoder/encoder/dense/bias/v
::8	?2)Adam/autoencoder/encoder/dense_1/kernel/v
3:12'Adam/autoencoder/encoder/dense_1/bias/v
::8	?2)Adam/autoencoder/decoder/dense_2/kernel/v
4:2?2'Adam/autoencoder/decoder/dense_2/bias/v
F:D22Adam/autoencoder/decoder/conv1d_transpose/kernel/v
<::20Adam/autoencoder/decoder/conv1d_transpose/bias/v
H:F&24Adam/autoencoder/decoder/conv1d_transpose_1/kernel/v
>:<&22Adam/autoencoder/decoder/conv1d_transpose_1/bias/v
?2?
+__inference_autoencoder_layer_call_fn_27374
+__inference_autoencoder_layer_call_fn_27574?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_autoencoder_layer_call_and_return_conditional_losses_27740
F__inference_autoencoder_layer_call_and_return_conditional_losses_27499?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
 __inference__wrapped_model_27022input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_encoder_layer_call_fn_27765?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_encoder_layer_call_and_return_conditional_losses_27830?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_decoder_layer_call_fn_27847?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_decoder_layer_call_and_return_conditional_losses_27943?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_27540input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_conv1d_transpose_layer_call_fn_27952?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_conv1d_transpose_layer_call_and_return_conditional_losses_27994?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_conv1d_transpose_1_layer_call_fn_28003?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_28045?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_27022 !"#$%&'()*+,-4?1
*?'
%?"
input_1?????????d&
? "7?4
2
output_1&?#
output_1?????????d&?
F__inference_autoencoder_layer_call_and_return_conditional_losses_27499 !"#$%&'()*+,-4?1
*?'
%?"
input_1?????????d&
? "7?4
?
0?????????d&
?
?	
1/0 ?
F__inference_autoencoder_layer_call_and_return_conditional_losses_27740~ !"#$%&'()*+,-3?0
)?&
$?!
inputs?????????d&
? "7?4
?
0?????????d&
?
?	
1/0 ?
+__inference_autoencoder_layer_call_fn_27374d !"#$%&'()*+,-4?1
*?'
%?"
input_1?????????d&
? "??????????d&?
+__inference_autoencoder_layer_call_fn_27574c !"#$%&'()*+,-3?0
)?&
$?!
inputs?????????d&
? "??????????d&?
M__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_28045v,-<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????&
? ?
2__inference_conv1d_transpose_1_layer_call_fn_28003i,-<?9
2?/
-?*
inputs??????????????????
? "%?"??????????????????&?
K__inference_conv1d_transpose_layer_call_and_return_conditional_losses_27994v*+<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????
? ?
0__inference_conv1d_transpose_layer_call_fn_27952i*+<?9
2?/
-?*
inputs??????????????????
? "%?"???????????????????
B__inference_decoder_layer_call_and_return_conditional_losses_27943d()*+,-/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????d&
? ?
'__inference_decoder_layer_call_fn_27847W()*+,-/?,
%?"
 ?
inputs?????????
? "??????????d&?
B__inference_encoder_layer_call_and_return_conditional_losses_27830? !"#$%&'3?0
)?&
$?!
inputs?????????d&
? "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
'__inference_encoder_layer_call_fn_27765? !"#$%&'3?0
)?&
$?!
inputs?????????d&
? "Z?W
?
0?????????
?
1?????????
?
2??????????
#__inference_signature_wrapper_27540? !"#$%&'()*+,-??<
? 
5?2
0
input_1%?"
input_1?????????d&"7?4
2
output_1&?#
output_1?????????d&