??
??
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??

z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:7$*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:7$*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:$*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:$* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:$*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:*
dtype0
?
conv1d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*(
shared_nameconv1d_transpose/kernel
?
+conv1d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose/kernel*"
_output_shapes
:$*
dtype0
?
conv1d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*&
shared_nameconv1d_transpose/bias
{
)conv1d_transpose/bias/Read/ReadVariableOpReadVariableOpconv1d_transpose/bias*
_output_shapes
:$*
dtype0
?
conv1d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:7$**
shared_nameconv1d_transpose_1/kernel
?
-conv1d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_transpose_1/kernel*"
_output_shapes
:7$*
dtype0
?
conv1d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*(
shared_nameconv1d_transpose_1/bias

+conv1d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv1d_transpose_1/bias*
_output_shapes
:7*
dtype0
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
Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7$*%
shared_nameAdam/conv1d/kernel/m
?
(Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/m*"
_output_shapes
:7$*
dtype0
|
Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*#
shared_nameAdam/conv1d/bias/m
u
&Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/m*
_output_shapes
:$*
dtype0
?
Adam/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*'
shared_nameAdam/conv1d_1/kernel/m
?
*Adam/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/m*"
_output_shapes
:$*
dtype0
?
Adam/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_1/bias/m
y
(Adam/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv1d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*/
shared_name Adam/conv1d_transpose/kernel/m
?
2Adam/conv1d_transpose/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_transpose/kernel/m*"
_output_shapes
:$*
dtype0
?
Adam/conv1d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*-
shared_nameAdam/conv1d_transpose/bias/m
?
0Adam/conv1d_transpose/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_transpose/bias/m*
_output_shapes
:$*
dtype0
?
 Adam/conv1d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7$*1
shared_name" Adam/conv1d_transpose_1/kernel/m
?
4Adam/conv1d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv1d_transpose_1/kernel/m*"
_output_shapes
:7$*
dtype0
?
Adam/conv1d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*/
shared_name Adam/conv1d_transpose_1/bias/m
?
2Adam/conv1d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_transpose_1/bias/m*
_output_shapes
:7*
dtype0
?
Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7$*%
shared_nameAdam/conv1d/kernel/v
?
(Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/v*"
_output_shapes
:7$*
dtype0
|
Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*#
shared_nameAdam/conv1d/bias/v
u
&Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/v*
_output_shapes
:$*
dtype0
?
Adam/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*'
shared_nameAdam/conv1d_1/kernel/v
?
*Adam/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/v*"
_output_shapes
:$*
dtype0
?
Adam/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_1/bias/v
y
(Adam/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv1d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*/
shared_name Adam/conv1d_transpose/kernel/v
?
2Adam/conv1d_transpose/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_transpose/kernel/v*"
_output_shapes
:$*
dtype0
?
Adam/conv1d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*-
shared_nameAdam/conv1d_transpose/bias/v
?
0Adam/conv1d_transpose/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_transpose/bias/v*
_output_shapes
:$*
dtype0
?
 Adam/conv1d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7$*1
shared_name" Adam/conv1d_transpose_1/kernel/v
?
4Adam/conv1d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv1d_transpose_1/kernel/v*"
_output_shapes
:7$*
dtype0
?
Adam/conv1d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*/
shared_name Adam/conv1d_transpose_1/bias/v
?
2Adam/conv1d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_transpose_1/bias/v*
_output_shapes
:7*
dtype0

NoOpNoOp
?,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?,
value?,B?, B?+
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
?
#iter

$beta_1

%beta_2
	&decay
'learning_ratemFmGmHmImJmKmLmMvNvOvPvQvRvSvTvU
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
 
?
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
 
YW
VARIABLE_VALUEconv1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
ca
VARIABLE_VALUEconv1d_transpose/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv1d_transpose/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
ec
VARIABLE_VALUEconv1d_transpose_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv1d_transpose_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
 trainable_variables
!regularization_losses
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
 

0
1
2
3

A0
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
4
	Btotal
	Ccount
D	variables
E	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

D	variables
|z
VARIABLE_VALUEAdam/conv1d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv1d_transpose/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv1d_transpose/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/conv1d_transpose_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv1d_transpose_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv1d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv1d_transpose/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv1d_transpose/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/conv1d_transpose_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv1d_transpose_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv1d_inputPlaceholder*+
_output_shapes
:?????????d7*
dtype0* 
shape:?????????d7
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_inputconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_transpose/kernelconv1d_transpose/biasconv1d_transpose_1/kernelconv1d_transpose_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d7**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_138284
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp+conv1d_transpose/kernel/Read/ReadVariableOp)conv1d_transpose/bias/Read/ReadVariableOp-conv1d_transpose_1/kernel/Read/ReadVariableOp+conv1d_transpose_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv1d/kernel/m/Read/ReadVariableOp&Adam/conv1d/bias/m/Read/ReadVariableOp*Adam/conv1d_1/kernel/m/Read/ReadVariableOp(Adam/conv1d_1/bias/m/Read/ReadVariableOp2Adam/conv1d_transpose/kernel/m/Read/ReadVariableOp0Adam/conv1d_transpose/bias/m/Read/ReadVariableOp4Adam/conv1d_transpose_1/kernel/m/Read/ReadVariableOp2Adam/conv1d_transpose_1/bias/m/Read/ReadVariableOp(Adam/conv1d/kernel/v/Read/ReadVariableOp&Adam/conv1d/bias/v/Read/ReadVariableOp*Adam/conv1d_1/kernel/v/Read/ReadVariableOp(Adam/conv1d_1/bias/v/Read/ReadVariableOp2Adam/conv1d_transpose/kernel/v/Read/ReadVariableOp0Adam/conv1d_transpose/bias/v/Read/ReadVariableOp4Adam/conv1d_transpose_1/kernel/v/Read/ReadVariableOp2Adam/conv1d_transpose_1/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_138904
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_transpose/kernelconv1d_transpose/biasconv1d_transpose_1/kernelconv1d_transpose_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv1d/kernel/mAdam/conv1d/bias/mAdam/conv1d_1/kernel/mAdam/conv1d_1/bias/mAdam/conv1d_transpose/kernel/mAdam/conv1d_transpose/bias/m Adam/conv1d_transpose_1/kernel/mAdam/conv1d_transpose_1/bias/mAdam/conv1d/kernel/vAdam/conv1d/bias/vAdam/conv1d_1/kernel/vAdam/conv1d_1/bias/vAdam/conv1d_transpose/kernel/vAdam/conv1d_transpose/bias/v Adam/conv1d_transpose_1/kernel/vAdam/conv1d_transpose_1/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_139007??	
?
?
D__inference_conv1d_1_layer_call_and_return_conditional_losses_138584

inputsA
+conv1d_expanddims_1_readvariableop_resource:$-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2$?
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:$?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????2$
 
_user_specified_nameinputs
?
?
D__inference_conv1d_1_layer_call_and_return_conditional_losses_137962

inputsA
+conv1d_expanddims_1_readvariableop_resource:$-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2$?
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:$?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????2$
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_138167

inputs#
conv1d_138146:7$
conv1d_138148:$%
conv1d_1_138151:$
conv1d_1_138153:-
conv1d_transpose_138156:$%
conv1d_transpose_138158:$/
conv1d_transpose_1_138161:7$'
conv1d_transpose_1_138163:7
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?(conv1d_transpose/StatefulPartitionedCall?*conv1d_transpose_1/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_138146conv1d_138148*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_137940?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_138151conv1d_1_138153*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_137962?
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_transpose_138156conv1d_transpose_138158*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_138009?
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0conv1d_transpose_1_138161conv1d_transpose_1_138163*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d7*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_138056?
IdentityIdentity3conv1d_transpose_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d7?
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d7: : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????d7
 
_user_specified_nameinputs
?	
?
+__inference_sequential_layer_call_fn_138326

inputs
unknown:7$
	unknown_0:$
	unknown_1:$
	unknown_2:
	unknown_3:$
	unknown_4:$
	unknown_5:7$
	unknown_6:7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d7**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_138167s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d7: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????d7
 
_user_specified_nameinputs
?	
?
$__inference_signature_wrapper_138284
conv1d_input
unknown:7$
	unknown_0:$
	unknown_1:$
	unknown_2:
	unknown_3:$
	unknown_4:$
	unknown_5:7$
	unknown_6:7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d7**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_137811s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d7: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????d7
&
_user_specified_nameconv1d_input
?+
?
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_138788

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:7$-
biasadd_readvariableop_resource:7
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
value	B :7n
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
T0*/
_output_shapes
:?????????2$?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
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
:7$n
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
:?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d7*
paddingVALID*
strides
?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*+
_output_shapes
:?????????d7*
squeeze_dims
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:7*
dtype0?
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d7T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????d7e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????d7?
NoOpNoOp^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????2$
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_138231
conv1d_input#
conv1d_138210:7$
conv1d_138212:$%
conv1d_1_138215:$
conv1d_1_138217:-
conv1d_transpose_138220:$%
conv1d_transpose_138222:$/
conv1d_transpose_1_138225:7$'
conv1d_transpose_1_138227:7
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?(conv1d_transpose/StatefulPartitionedCall?*conv1d_transpose_1/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallconv1d_inputconv1d_138210conv1d_138212*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_137940?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_138215conv1d_1_138217*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_137962?
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_transpose_138220conv1d_transpose_138222*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_138009?
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0conv1d_transpose_1_138225conv1d_transpose_1_138227*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d7*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_138056?
IdentityIdentity3conv1d_transpose_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d7?
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d7: : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????d7
&
_user_specified_nameconv1d_input
?+
?
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_138056

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:7$-
biasadd_readvariableop_resource:7
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
value	B :7n
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
T0*/
_output_shapes
:?????????2$?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
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
:7$n
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
:?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d7*
paddingVALID*
strides
?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*+
_output_shapes
:?????????d7*
squeeze_dims
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:7*
dtype0?
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d7T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????d7e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????d7?
NoOpNoOp^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????2$
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_138255
conv1d_input#
conv1d_138234:7$
conv1d_138236:$%
conv1d_1_138239:$
conv1d_1_138241:-
conv1d_transpose_138244:$%
conv1d_transpose_138246:$/
conv1d_transpose_1_138249:7$'
conv1d_transpose_1_138251:7
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?(conv1d_transpose/StatefulPartitionedCall?*conv1d_transpose_1/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallconv1d_inputconv1d_138234conv1d_138236*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_137940?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_138239conv1d_1_138241*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_137962?
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_transpose_138244conv1d_transpose_138246*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_138009?
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0conv1d_transpose_1_138249conv1d_transpose_1_138251*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d7*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_138056?
IdentityIdentity3conv1d_transpose_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d7?
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d7: : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????d7
&
_user_specified_nameconv1d_input
??
?	
!__inference__wrapped_model_137811
conv1d_inputS
=sequential_conv1d_conv1d_expanddims_1_readvariableop_resource:7$?
1sequential_conv1d_biasadd_readvariableop_resource:$U
?sequential_conv1d_1_conv1d_expanddims_1_readvariableop_resource:$A
3sequential_conv1d_1_biasadd_readvariableop_resource:g
Qsequential_conv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource:$I
;sequential_conv1d_transpose_biasadd_readvariableop_resource:$i
Ssequential_conv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource:7$K
=sequential_conv1d_transpose_1_biasadd_readvariableop_resource:7
identity??(sequential/conv1d/BiasAdd/ReadVariableOp?4sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp?*sequential/conv1d_1/BiasAdd/ReadVariableOp?6sequential/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp?2sequential/conv1d_transpose/BiasAdd/ReadVariableOp?Hsequential/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?4sequential/conv1d_transpose_1/BiasAdd/ReadVariableOp?Jsequential/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpr
'sequential/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#sequential/conv1d/Conv1D/ExpandDims
ExpandDimsconv1d_input0sequential/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d7?
4sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=sequential_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
dtype0k
)sequential/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/conv1d/Conv1D/ExpandDims_1
ExpandDims<sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:02sequential/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:7$?
sequential/conv1d/Conv1DConv2D,sequential/conv1d/Conv1D/ExpandDims:output:0.sequential/conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????2$*
paddingVALID*
strides
?
 sequential/conv1d/Conv1D/SqueezeSqueeze!sequential/conv1d/Conv1D:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims

??????????
(sequential/conv1d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv1d_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
sequential/conv1d/BiasAddBiasAdd)sequential/conv1d/Conv1D/Squeeze:output:00sequential/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$x
sequential/conv1d/ReluRelu"sequential/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$t
)sequential/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%sequential/conv1d_1/Conv1D/ExpandDims
ExpandDims$sequential/conv1d/Relu:activations:02sequential/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2$?
6sequential/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp?sequential_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
dtype0m
+sequential/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential/conv1d_1/Conv1D/ExpandDims_1
ExpandDims>sequential/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:04sequential/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:$?
sequential/conv1d_1/Conv1DConv2D.sequential/conv1d_1/Conv1D/ExpandDims:output:00sequential/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
"sequential/conv1d_1/Conv1D/SqueezeSqueeze#sequential/conv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
*sequential/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/conv1d_1/BiasAddBiasAdd+sequential/conv1d_1/Conv1D/Squeeze:output:02sequential/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????|
sequential/conv1d_1/ReluRelu$sequential/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????w
!sequential/conv1d_transpose/ShapeShape&sequential/conv1d_1/Relu:activations:0*
T0*
_output_shapes
:y
/sequential/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential/conv1d_transpose/strided_sliceStridedSlice*sequential/conv1d_transpose/Shape:output:08sequential/conv1d_transpose/strided_slice/stack:output:0:sequential/conv1d_transpose/strided_slice/stack_1:output:0:sequential/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1sequential/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3sequential/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+sequential/conv1d_transpose/strided_slice_1StridedSlice*sequential/conv1d_transpose/Shape:output:0:sequential/conv1d_transpose/strided_slice_1/stack:output:0<sequential/conv1d_transpose/strided_slice_1/stack_1:output:0<sequential/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!sequential/conv1d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential/conv1d_transpose/mulMul4sequential/conv1d_transpose/strided_slice_1:output:0*sequential/conv1d_transpose/mul/y:output:0*
T0*
_output_shapes
: c
!sequential/conv1d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential/conv1d_transpose/addAddV2#sequential/conv1d_transpose/mul:z:0*sequential/conv1d_transpose/add/y:output:0*
T0*
_output_shapes
: e
#sequential/conv1d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :$?
!sequential/conv1d_transpose/stackPack2sequential/conv1d_transpose/strided_slice:output:0#sequential/conv1d_transpose/add:z:0,sequential/conv1d_transpose/stack/2:output:0*
N*
T0*
_output_shapes
:}
;sequential/conv1d_transpose/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
7sequential/conv1d_transpose/conv1d_transpose/ExpandDims
ExpandDims&sequential/conv1d_1/Relu:activations:0Dsequential/conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
Hsequential/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpQsequential_conv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
dtype0
=sequential/conv1d_transpose/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
9sequential/conv1d_transpose/conv1d_transpose/ExpandDims_1
ExpandDimsPsequential/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Fsequential/conv1d_transpose/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:$?
@sequential/conv1d_transpose/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bsequential/conv1d_transpose/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bsequential/conv1d_transpose/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:sequential/conv1d_transpose/conv1d_transpose/strided_sliceStridedSlice*sequential/conv1d_transpose/stack:output:0Isequential/conv1d_transpose/conv1d_transpose/strided_slice/stack:output:0Ksequential/conv1d_transpose/conv1d_transpose/strided_slice/stack_1:output:0Ksequential/conv1d_transpose/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Bsequential/conv1d_transpose/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Dsequential/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential/conv1d_transpose/conv1d_transpose/strided_slice_1StridedSlice*sequential/conv1d_transpose/stack:output:0Ksequential/conv1d_transpose/conv1d_transpose/strided_slice_1/stack:output:0Msequential/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1:output:0Msequential/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
<sequential/conv1d_transpose/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:z
8sequential/conv1d_transpose/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3sequential/conv1d_transpose/conv1d_transpose/concatConcatV2Csequential/conv1d_transpose/conv1d_transpose/strided_slice:output:0Esequential/conv1d_transpose/conv1d_transpose/concat/values_1:output:0Esequential/conv1d_transpose/conv1d_transpose/strided_slice_1:output:0Asequential/conv1d_transpose/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,sequential/conv1d_transpose/conv1d_transposeConv2DBackpropInput<sequential/conv1d_transpose/conv1d_transpose/concat:output:0Bsequential/conv1d_transpose/conv1d_transpose/ExpandDims_1:output:0@sequential/conv1d_transpose/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????2$*
paddingVALID*
strides
?
4sequential/conv1d_transpose/conv1d_transpose/SqueezeSqueeze5sequential/conv1d_transpose/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims
?
2sequential/conv1d_transpose/BiasAdd/ReadVariableOpReadVariableOp;sequential_conv1d_transpose_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
#sequential/conv1d_transpose/BiasAddBiasAdd=sequential/conv1d_transpose/conv1d_transpose/Squeeze:output:0:sequential/conv1d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$?
 sequential/conv1d_transpose/ReluRelu,sequential/conv1d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$?
#sequential/conv1d_transpose_1/ShapeShape.sequential/conv1d_transpose/Relu:activations:0*
T0*
_output_shapes
:{
1sequential/conv1d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential/conv1d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential/conv1d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+sequential/conv1d_transpose_1/strided_sliceStridedSlice,sequential/conv1d_transpose_1/Shape:output:0:sequential/conv1d_transpose_1/strided_slice/stack:output:0<sequential/conv1d_transpose_1/strided_slice/stack_1:output:0<sequential/conv1d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
3sequential/conv1d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
5sequential/conv1d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential/conv1d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-sequential/conv1d_transpose_1/strided_slice_1StridedSlice,sequential/conv1d_transpose_1/Shape:output:0<sequential/conv1d_transpose_1/strided_slice_1/stack:output:0>sequential/conv1d_transpose_1/strided_slice_1/stack_1:output:0>sequential/conv1d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential/conv1d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
!sequential/conv1d_transpose_1/mulMul6sequential/conv1d_transpose_1/strided_slice_1:output:0,sequential/conv1d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: e
#sequential/conv1d_transpose_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
!sequential/conv1d_transpose_1/addAddV2%sequential/conv1d_transpose_1/mul:z:0,sequential/conv1d_transpose_1/add/y:output:0*
T0*
_output_shapes
: g
%sequential/conv1d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :7?
#sequential/conv1d_transpose_1/stackPack4sequential/conv1d_transpose_1/strided_slice:output:0%sequential/conv1d_transpose_1/add:z:0.sequential/conv1d_transpose_1/stack/2:output:0*
N*
T0*
_output_shapes
:
=sequential/conv1d_transpose_1/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
9sequential/conv1d_transpose_1/conv1d_transpose/ExpandDims
ExpandDims.sequential/conv1d_transpose/Relu:activations:0Fsequential/conv1d_transpose_1/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2$?
Jsequential/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpSsequential_conv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
dtype0?
?sequential/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
;sequential/conv1d_transpose_1/conv1d_transpose/ExpandDims_1
ExpandDimsRsequential/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Hsequential/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:7$?
Bsequential/conv1d_transpose_1/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Dsequential/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential/conv1d_transpose_1/conv1d_transpose/strided_sliceStridedSlice,sequential/conv1d_transpose_1/stack:output:0Ksequential/conv1d_transpose_1/conv1d_transpose/strided_slice/stack:output:0Msequential/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1:output:0Msequential/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Dsequential/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Fsequential/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Fsequential/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
>sequential/conv1d_transpose_1/conv1d_transpose/strided_slice_1StridedSlice,sequential/conv1d_transpose_1/stack:output:0Msequential/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack:output:0Osequential/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1:output:0Osequential/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
>sequential/conv1d_transpose_1/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:|
:sequential/conv1d_transpose_1/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5sequential/conv1d_transpose_1/conv1d_transpose/concatConcatV2Esequential/conv1d_transpose_1/conv1d_transpose/strided_slice:output:0Gsequential/conv1d_transpose_1/conv1d_transpose/concat/values_1:output:0Gsequential/conv1d_transpose_1/conv1d_transpose/strided_slice_1:output:0Csequential/conv1d_transpose_1/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
.sequential/conv1d_transpose_1/conv1d_transposeConv2DBackpropInput>sequential/conv1d_transpose_1/conv1d_transpose/concat:output:0Dsequential/conv1d_transpose_1/conv1d_transpose/ExpandDims_1:output:0Bsequential/conv1d_transpose_1/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d7*
paddingVALID*
strides
?
6sequential/conv1d_transpose_1/conv1d_transpose/SqueezeSqueeze7sequential/conv1d_transpose_1/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????d7*
squeeze_dims
?
4sequential/conv1d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp=sequential_conv1d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0?
%sequential/conv1d_transpose_1/BiasAddBiasAdd?sequential/conv1d_transpose_1/conv1d_transpose/Squeeze:output:0<sequential/conv1d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d7?
"sequential/conv1d_transpose_1/ReluRelu.sequential/conv1d_transpose_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d7?
IdentityIdentity0sequential/conv1d_transpose_1/Relu:activations:0^NoOp*
T0*+
_output_shapes
:?????????d7?
NoOpNoOp)^sequential/conv1d/BiasAdd/ReadVariableOp5^sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp+^sequential/conv1d_1/BiasAdd/ReadVariableOp7^sequential/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp3^sequential/conv1d_transpose/BiasAdd/ReadVariableOpI^sequential/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp5^sequential/conv1d_transpose_1/BiasAdd/ReadVariableOpK^sequential/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d7: : : : : : : : 2T
(sequential/conv1d/BiasAdd/ReadVariableOp(sequential/conv1d/BiasAdd/ReadVariableOp2l
4sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp4sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2X
*sequential/conv1d_1/BiasAdd/ReadVariableOp*sequential/conv1d_1/BiasAdd/ReadVariableOp2p
6sequential/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp6sequential/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2h
2sequential/conv1d_transpose/BiasAdd/ReadVariableOp2sequential/conv1d_transpose/BiasAdd/ReadVariableOp2?
Hsequential/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpHsequential/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2l
4sequential/conv1d_transpose_1/BiasAdd/ReadVariableOp4sequential/conv1d_transpose_1/BiasAdd/ReadVariableOp2?
Jsequential/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpJsequential/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp:Y U
+
_output_shapes
:?????????d7
&
_user_specified_nameconv1d_input
?~
?
"__inference__traced_restore_139007
file_prefix4
assignvariableop_conv1d_kernel:7$,
assignvariableop_1_conv1d_bias:$8
"assignvariableop_2_conv1d_1_kernel:$.
 assignvariableop_3_conv1d_1_bias:@
*assignvariableop_4_conv1d_transpose_kernel:$6
(assignvariableop_5_conv1d_transpose_bias:$B
,assignvariableop_6_conv1d_transpose_1_kernel:7$8
*assignvariableop_7_conv1d_transpose_1_bias:7&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: >
(assignvariableop_15_adam_conv1d_kernel_m:7$4
&assignvariableop_16_adam_conv1d_bias_m:$@
*assignvariableop_17_adam_conv1d_1_kernel_m:$6
(assignvariableop_18_adam_conv1d_1_bias_m:H
2assignvariableop_19_adam_conv1d_transpose_kernel_m:$>
0assignvariableop_20_adam_conv1d_transpose_bias_m:$J
4assignvariableop_21_adam_conv1d_transpose_1_kernel_m:7$@
2assignvariableop_22_adam_conv1d_transpose_1_bias_m:7>
(assignvariableop_23_adam_conv1d_kernel_v:7$4
&assignvariableop_24_adam_conv1d_bias_v:$@
*assignvariableop_25_adam_conv1d_1_kernel_v:$6
(assignvariableop_26_adam_conv1d_1_bias_v:H
2assignvariableop_27_adam_conv1d_transpose_kernel_v:$>
0assignvariableop_28_adam_conv1d_transpose_bias_v:$J
4assignvariableop_29_adam_conv1d_transpose_1_kernel_v:7$@
2assignvariableop_30_adam_conv1d_transpose_1_bias_v:7
identity_32??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp*assignvariableop_4_conv1d_transpose_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp(assignvariableop_5_conv1d_transpose_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp,assignvariableop_6_conv1d_transpose_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp*assignvariableop_7_conv1d_transpose_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_conv1d_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_conv1d_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_conv1d_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_conv1d_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp2assignvariableop_19_adam_conv1d_transpose_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_adam_conv1d_transpose_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_conv1d_transpose_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp2assignvariableop_22_adam_conv1d_transpose_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_conv1d_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_conv1d_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv1d_1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv1d_1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_conv1d_transpose_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp0assignvariableop_28_adam_conv1d_transpose_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_conv1d_transpose_1_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_conv1d_transpose_1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
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
܊
?
F__inference_sequential_layer_call_and_return_conditional_losses_138534

inputsH
2conv1d_conv1d_expanddims_1_readvariableop_resource:7$4
&conv1d_biasadd_readvariableop_resource:$J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:$6
(conv1d_1_biasadd_readvariableop_resource:\
Fconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource:$>
0conv1d_transpose_biasadd_readvariableop_resource:$^
Hconv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource:7$@
2conv1d_transpose_1_biasadd_readvariableop_resource:7
identity??conv1d/BiasAdd/ReadVariableOp?)conv1d/Conv1D/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp?'conv1d_transpose/BiasAdd/ReadVariableOp?=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?)conv1d_transpose_1/BiasAdd/ReadVariableOp??conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpg
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
:?????????d7?
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
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
:7$?
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????2$*
paddingVALID*
strides
?
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims

??????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$b
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$i
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
:?????????2$?
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
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
:$?
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????f
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????a
conv1d_transpose/ShapeShapeconv1d_1/Relu:activations:0*
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
value	B :$?
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
ExpandDimsconv1d_1/Relu:activations:09conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpFconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
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
:$
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
:?????????2$*
paddingVALID*
strides
?
)conv1d_transpose/conv1d_transpose/SqueezeSqueeze*conv1d_transpose/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims
?
'conv1d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv1d_transpose_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
conv1d_transpose/BiasAddBiasAdd2conv1d_transpose/conv1d_transpose/Squeeze:output:0/conv1d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$v
conv1d_transpose/ReluRelu!conv1d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$k
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
value	B :7?
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
:?????????2$?
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
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
:7$?
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
:?????????d7*
paddingVALID*
strides
?
+conv1d_transpose_1/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_1/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????d7*
squeeze_dims
?
)conv1d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0?
conv1d_transpose_1/BiasAddBiasAdd4conv1d_transpose_1/conv1d_transpose/Squeeze:output:01conv1d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d7z
conv1d_transpose_1/ReluRelu#conv1d_transpose_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d7x
IdentityIdentity%conv1d_transpose_1/Relu:activations:0^NoOp*
T0*+
_output_shapes
:?????????d7?
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp(^conv1d_transpose/BiasAdd/ReadVariableOp>^conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp*^conv1d_transpose_1/BiasAdd/ReadVariableOp@^conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d7: : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2R
'conv1d_transpose/BiasAdd/ReadVariableOp'conv1d_transpose/BiasAdd/ReadVariableOp2~
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2V
)conv1d_transpose_1/BiasAdd/ReadVariableOp)conv1d_transpose_1/BiasAdd/ReadVariableOp2?
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????d7
 
_user_specified_nameinputs
܊
?
F__inference_sequential_layer_call_and_return_conditional_losses_138430

inputsH
2conv1d_conv1d_expanddims_1_readvariableop_resource:7$4
&conv1d_biasadd_readvariableop_resource:$J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:$6
(conv1d_1_biasadd_readvariableop_resource:\
Fconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource:$>
0conv1d_transpose_biasadd_readvariableop_resource:$^
Hconv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource:7$@
2conv1d_transpose_1_biasadd_readvariableop_resource:7
identity??conv1d/BiasAdd/ReadVariableOp?)conv1d/Conv1D/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp?'conv1d_transpose/BiasAdd/ReadVariableOp?=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp?)conv1d_transpose_1/BiasAdd/ReadVariableOp??conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpg
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
:?????????d7?
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
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
:7$?
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????2$*
paddingVALID*
strides
?
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims

??????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$b
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$i
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
:?????????2$?
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
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
:$?
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????f
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????a
conv1d_transpose/ShapeShapeconv1d_1/Relu:activations:0*
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
value	B :$?
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
ExpandDimsconv1d_1/Relu:activations:09conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpFconv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
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
:$
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
:?????????2$*
paddingVALID*
strides
?
)conv1d_transpose/conv1d_transpose/SqueezeSqueeze*conv1d_transpose/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims
?
'conv1d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv1d_transpose_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
conv1d_transpose/BiasAddBiasAdd2conv1d_transpose/conv1d_transpose/Squeeze:output:0/conv1d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$v
conv1d_transpose/ReluRelu!conv1d_transpose/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$k
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
value	B :7?
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
:?????????2$?
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
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
:7$?
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
:?????????d7*
paddingVALID*
strides
?
+conv1d_transpose_1/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_1/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????d7*
squeeze_dims
?
)conv1d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0?
conv1d_transpose_1/BiasAddBiasAdd4conv1d_transpose_1/conv1d_transpose/Squeeze:output:01conv1d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d7z
conv1d_transpose_1/ReluRelu#conv1d_transpose_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d7x
IdentityIdentity%conv1d_transpose_1/Relu:activations:0^NoOp*
T0*+
_output_shapes
:?????????d7?
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp(^conv1d_transpose/BiasAdd/ReadVariableOp>^conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp*^conv1d_transpose_1/BiasAdd/ReadVariableOp@^conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d7: : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2R
'conv1d_transpose/BiasAdd/ReadVariableOp'conv1d_transpose/BiasAdd/ReadVariableOp2~
=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp=conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2V
)conv1d_transpose_1/BiasAdd/ReadVariableOp)conv1d_transpose_1/BiasAdd/ReadVariableOp2?
?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????d7
 
_user_specified_nameinputs
?
?
3__inference_conv1d_transpose_1_layer_call_fn_138695

inputs
unknown:7$
	unknown_0:7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????7*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_137910|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????$: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????$
 
_user_specified_nameinputs
?
?
1__inference_conv1d_transpose_layer_call_fn_138593

inputs
unknown:$
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_137857|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
+__inference_sequential_layer_call_fn_138082
conv1d_input
unknown:7$
	unknown_0:$
	unknown_1:$
	unknown_2:
	unknown_3:$
	unknown_4:$
	unknown_5:7$
	unknown_6:7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d7**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_138063s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d7: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????d7
&
_user_specified_nameconv1d_input
?
?
B__inference_conv1d_layer_call_and_return_conditional_losses_138559

inputsA
+conv1d_expanddims_1_readvariableop_resource:7$-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d7?
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:7$?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????2$*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims

?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????2$e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????2$?
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d7: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????d7
 
_user_specified_nameinputs
?
?
1__inference_conv1d_transpose_layer_call_fn_138602

inputs
unknown:$
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_138009s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?+
?
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_138686

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:$-
biasadd_readvariableop_resource:$
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
value	B :$n
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
T0*/
_output_shapes
:??????????
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
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
:$n
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
:?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????2$*
paddingVALID*
strides
?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????2$e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????2$?
NoOpNoOp^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_conv1d_1_layer_call_fn_138568

inputs
unknown:$
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_137962s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2$: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????2$
 
_user_specified_nameinputs
?	
?
+__inference_sequential_layer_call_fn_138305

inputs
unknown:7$
	unknown_0:$
	unknown_1:$
	unknown_2:
	unknown_3:$
	unknown_4:$
	unknown_5:7$
	unknown_6:7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d7**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_138063s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d7: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????d7
 
_user_specified_nameinputs
?	
?
+__inference_sequential_layer_call_fn_138207
conv1d_input
unknown:7$
	unknown_0:$
	unknown_1:$
	unknown_2:
	unknown_3:$
	unknown_4:$
	unknown_5:7$
	unknown_6:7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d7**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_138167s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d7: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????d7
&
_user_specified_nameconv1d_input
?,
?
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_137857

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:$-
biasadd_readvariableop_resource:$
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
value	B :$n
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
$:"???????????????????
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
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
:$n
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
$:"??????????????????$*
paddingVALID*
strides
?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????$*
squeeze_dims
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????$]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????$n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????$?
NoOpNoOp^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?,
?
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_138644

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:$-
biasadd_readvariableop_resource:$
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
value	B :$n
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
$:"???????????????????
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
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
:$n
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
$:"??????????????????$*
paddingVALID*
strides
?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????$*
squeeze_dims
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????$]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????$n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????$?
NoOpNoOp^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
3__inference_conv1d_transpose_1_layer_call_fn_138704

inputs
unknown:7$
	unknown_0:7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d7*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_138056s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2$: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????2$
 
_user_specified_nameinputs
?+
?
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_138009

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:$-
biasadd_readvariableop_resource:$
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
value	B :$n
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
T0*/
_output_shapes
:??????????
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
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
:$n
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
:?
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????2$*
paddingVALID*
strides
?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????2$e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????2$?
NoOpNoOp^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?,
?
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_138746

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:7$-
biasadd_readvariableop_resource:7
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
value	B :7n
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
$:"??????????????????$?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
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
:7$n
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
$:"??????????????????7*
paddingVALID*
strides
?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????7*
squeeze_dims
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:7*
dtype0?
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????7]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????7n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????7?
NoOpNoOp^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????$
 
_user_specified_nameinputs
?E
?
__inference__traced_save_138904
file_prefix,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop6
2savev2_conv1d_transpose_kernel_read_readvariableop4
0savev2_conv1d_transpose_bias_read_readvariableop8
4savev2_conv1d_transpose_1_kernel_read_readvariableop6
2savev2_conv1d_transpose_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv1d_kernel_m_read_readvariableop1
-savev2_adam_conv1d_bias_m_read_readvariableop5
1savev2_adam_conv1d_1_kernel_m_read_readvariableop3
/savev2_adam_conv1d_1_bias_m_read_readvariableop=
9savev2_adam_conv1d_transpose_kernel_m_read_readvariableop;
7savev2_adam_conv1d_transpose_bias_m_read_readvariableop?
;savev2_adam_conv1d_transpose_1_kernel_m_read_readvariableop=
9savev2_adam_conv1d_transpose_1_bias_m_read_readvariableop3
/savev2_adam_conv1d_kernel_v_read_readvariableop1
-savev2_adam_conv1d_bias_v_read_readvariableop5
1savev2_adam_conv1d_1_kernel_v_read_readvariableop3
/savev2_adam_conv1d_1_bias_v_read_readvariableop=
9savev2_adam_conv1d_transpose_kernel_v_read_readvariableop;
7savev2_adam_conv1d_transpose_bias_v_read_readvariableop?
;savev2_adam_conv1d_transpose_1_kernel_v_read_readvariableop=
9savev2_adam_conv1d_transpose_1_bias_v_read_readvariableop
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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop2savev2_conv1d_transpose_kernel_read_readvariableop0savev2_conv1d_transpose_bias_read_readvariableop4savev2_conv1d_transpose_1_kernel_read_readvariableop2savev2_conv1d_transpose_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv1d_kernel_m_read_readvariableop-savev2_adam_conv1d_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop9savev2_adam_conv1d_transpose_kernel_m_read_readvariableop7savev2_adam_conv1d_transpose_bias_m_read_readvariableop;savev2_adam_conv1d_transpose_1_kernel_m_read_readvariableop9savev2_adam_conv1d_transpose_1_bias_m_read_readvariableop/savev2_adam_conv1d_kernel_v_read_readvariableop-savev2_adam_conv1d_bias_v_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop9savev2_adam_conv1d_transpose_kernel_v_read_readvariableop7savev2_adam_conv1d_transpose_bias_v_read_readvariableop;savev2_adam_conv1d_transpose_1_kernel_v_read_readvariableop9savev2_adam_conv1d_transpose_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :7$:$:$::$:$:7$:7: : : : : : : :7$:$:$::$:$:7$:7:7$:$:$::$:$:7$:7: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:7$: 

_output_shapes
:$:($
"
_output_shapes
:$: 

_output_shapes
::($
"
_output_shapes
:$: 

_output_shapes
:$:($
"
_output_shapes
:7$: 

_output_shapes
:7:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:7$: 

_output_shapes
:$:($
"
_output_shapes
:$: 

_output_shapes
::($
"
_output_shapes
:$: 

_output_shapes
:$:($
"
_output_shapes
:7$: 

_output_shapes
:7:($
"
_output_shapes
:7$: 

_output_shapes
:$:($
"
_output_shapes
:$: 

_output_shapes
::($
"
_output_shapes
:$: 

_output_shapes
:$:($
"
_output_shapes
:7$: 

_output_shapes
:7: 

_output_shapes
: 
?
?
'__inference_conv1d_layer_call_fn_138543

inputs
unknown:7$
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_137940s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d7: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????d7
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_138063

inputs#
conv1d_137941:7$
conv1d_137943:$%
conv1d_1_137963:$
conv1d_1_137965:-
conv1d_transpose_138010:$%
conv1d_transpose_138012:$/
conv1d_transpose_1_138057:7$'
conv1d_transpose_1_138059:7
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?(conv1d_transpose/StatefulPartitionedCall?*conv1d_transpose_1/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_137941conv1d_137943*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_137940?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_137963conv1d_1_137965*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_137962?
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_transpose_138010conv1d_transpose_138012*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_138009?
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0conv1d_transpose_1_138057conv1d_transpose_1_138059*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d7*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_138056?
IdentityIdentity3conv1d_transpose_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d7?
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d7: : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????d7
 
_user_specified_nameinputs
?
?
B__inference_conv1d_layer_call_and_return_conditional_losses_137940

inputsA
+conv1d_expanddims_1_readvariableop_resource:7$-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d7?
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:7$?
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????2$*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims

?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????2$e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????2$?
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d7: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????d7
 
_user_specified_nameinputs
?,
?
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_137910

inputsK
5conv1d_transpose_expanddims_1_readvariableop_resource:7$-
biasadd_readvariableop_resource:7
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
value	B :7n
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
$:"??????????????????$?
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
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
:7$n
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
$:"??????????????????7*
paddingVALID*
strides
?
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :??????????????????7*
squeeze_dims
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:7*
dtype0?
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????7]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????7n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????7?
NoOpNoOp^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????$
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
conv1d_input9
serving_default_conv1d_input:0?????????d7J
conv1d_transpose_14
StatefulPartitionedCall:0?????????d7tensorflow/serving/predict:?f
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
V__call__
*W&call_and_return_all_conditional_losses
X_default_save_signature"
_tf_keras_sequential
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#iter

$beta_1

%beta_2
	&decay
'learning_ratemFmGmHmImJmKmLmMvNvOvPvQvRvSvTvU"
	optimizer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
V__call__
X_default_save_signature
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
,
aserving_default"
signature_map
#:!7$2conv1d/kernel
:$2conv1d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
%:#$2conv1d_1/kernel
:2conv1d_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
-:+$2conv1d_transpose/kernel
#:!$2conv1d_transpose/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
/:-7$2conv1d_transpose_1/kernel
%:#72conv1d_transpose_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
 trainable_variables
!regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
A0"
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
N
	Btotal
	Ccount
D	variables
E	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
B0
C1"
trackable_list_wrapper
-
D	variables"
_generic_user_object
(:&7$2Adam/conv1d/kernel/m
:$2Adam/conv1d/bias/m
*:($2Adam/conv1d_1/kernel/m
 :2Adam/conv1d_1/bias/m
2:0$2Adam/conv1d_transpose/kernel/m
(:&$2Adam/conv1d_transpose/bias/m
4:27$2 Adam/conv1d_transpose_1/kernel/m
*:(72Adam/conv1d_transpose_1/bias/m
(:&7$2Adam/conv1d/kernel/v
:$2Adam/conv1d/bias/v
*:($2Adam/conv1d_1/kernel/v
 :2Adam/conv1d_1/bias/v
2:0$2Adam/conv1d_transpose/kernel/v
(:&$2Adam/conv1d_transpose/bias/v
4:27$2 Adam/conv1d_transpose_1/kernel/v
*:(72Adam/conv1d_transpose_1/bias/v
?2?
+__inference_sequential_layer_call_fn_138082
+__inference_sequential_layer_call_fn_138305
+__inference_sequential_layer_call_fn_138326
+__inference_sequential_layer_call_fn_138207?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_sequential_layer_call_and_return_conditional_losses_138430
F__inference_sequential_layer_call_and_return_conditional_losses_138534
F__inference_sequential_layer_call_and_return_conditional_losses_138231
F__inference_sequential_layer_call_and_return_conditional_losses_138255?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_137811conv1d_input"?
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
'__inference_conv1d_layer_call_fn_138543?
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
B__inference_conv1d_layer_call_and_return_conditional_losses_138559?
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
)__inference_conv1d_1_layer_call_fn_138568?
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
D__inference_conv1d_1_layer_call_and_return_conditional_losses_138584?
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
1__inference_conv1d_transpose_layer_call_fn_138593
1__inference_conv1d_transpose_layer_call_fn_138602?
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
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_138644
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_138686?
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
3__inference_conv1d_transpose_1_layer_call_fn_138695
3__inference_conv1d_transpose_1_layer_call_fn_138704?
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
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_138746
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_138788?
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
$__inference_signature_wrapper_138284conv1d_input"?
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
 ?
!__inference__wrapped_model_137811?9?6
/?,
*?'
conv1d_input?????????d7
? "K?H
F
conv1d_transpose_10?-
conv1d_transpose_1?????????d7?
D__inference_conv1d_1_layer_call_and_return_conditional_losses_138584d3?0
)?&
$?!
inputs?????????2$
? ")?&
?
0?????????
? ?
)__inference_conv1d_1_layer_call_fn_138568W3?0
)?&
$?!
inputs?????????2$
? "???????????
B__inference_conv1d_layer_call_and_return_conditional_losses_138559d3?0
)?&
$?!
inputs?????????d7
? ")?&
?
0?????????2$
? ?
'__inference_conv1d_layer_call_fn_138543W3?0
)?&
$?!
inputs?????????d7
? "??????????2$?
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_138746v<?9
2?/
-?*
inputs??????????????????$
? "2?/
(?%
0??????????????????7
? ?
N__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_138788d3?0
)?&
$?!
inputs?????????2$
? ")?&
?
0?????????d7
? ?
3__inference_conv1d_transpose_1_layer_call_fn_138695i<?9
2?/
-?*
inputs??????????????????$
? "%?"??????????????????7?
3__inference_conv1d_transpose_1_layer_call_fn_138704W3?0
)?&
$?!
inputs?????????2$
? "??????????d7?
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_138644v<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????$
? ?
L__inference_conv1d_transpose_layer_call_and_return_conditional_losses_138686d3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????2$
? ?
1__inference_conv1d_transpose_layer_call_fn_138593i<?9
2?/
-?*
inputs??????????????????
? "%?"??????????????????$?
1__inference_conv1d_transpose_layer_call_fn_138602W3?0
)?&
$?!
inputs?????????
? "??????????2$?
F__inference_sequential_layer_call_and_return_conditional_losses_138231xA?>
7?4
*?'
conv1d_input?????????d7
p 

 
? ")?&
?
0?????????d7
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_138255xA?>
7?4
*?'
conv1d_input?????????d7
p

 
? ")?&
?
0?????????d7
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_138430r;?8
1?.
$?!
inputs?????????d7
p 

 
? ")?&
?
0?????????d7
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_138534r;?8
1?.
$?!
inputs?????????d7
p

 
? ")?&
?
0?????????d7
? ?
+__inference_sequential_layer_call_fn_138082kA?>
7?4
*?'
conv1d_input?????????d7
p 

 
? "??????????d7?
+__inference_sequential_layer_call_fn_138207kA?>
7?4
*?'
conv1d_input?????????d7
p

 
? "??????????d7?
+__inference_sequential_layer_call_fn_138305e;?8
1?.
$?!
inputs?????????d7
p 

 
? "??????????d7?
+__inference_sequential_layer_call_fn_138326e;?8
1?.
$?!
inputs?????????d7
p

 
? "??????????d7?
$__inference_signature_wrapper_138284?I?F
? 
??<
:
conv1d_input*?'
conv1d_input?????????d7"K?H
F
conv1d_transpose_10?-
conv1d_transpose_1?????????d7