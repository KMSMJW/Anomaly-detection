??
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
?
#autoencoder/encoder/conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:7$*4
shared_name%#autoencoder/encoder/conv1d_2/kernel
?
7autoencoder/encoder/conv1d_2/kernel/Read/ReadVariableOpReadVariableOp#autoencoder/encoder/conv1d_2/kernel*"
_output_shapes
:7$*
dtype0
?
!autoencoder/encoder/conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*2
shared_name#!autoencoder/encoder/conv1d_2/bias
?
5autoencoder/encoder/conv1d_2/bias/Read/ReadVariableOpReadVariableOp!autoencoder/encoder/conv1d_2/bias*
_output_shapes
:$*
dtype0
?
#autoencoder/encoder/conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*4
shared_name%#autoencoder/encoder/conv1d_3/kernel
?
7autoencoder/encoder/conv1d_3/kernel/Read/ReadVariableOpReadVariableOp#autoencoder/encoder/conv1d_3/kernel*"
_output_shapes
:$*
dtype0
?
!autoencoder/encoder/conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!autoencoder/encoder/conv1d_3/bias
?
5autoencoder/encoder/conv1d_3/bias/Read/ReadVariableOpReadVariableOp!autoencoder/encoder/conv1d_3/bias*
_output_shapes
:*
dtype0
?
"autoencoder/encoder/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"autoencoder/encoder/dense_3/kernel
?
6autoencoder/encoder/dense_3/kernel/Read/ReadVariableOpReadVariableOp"autoencoder/encoder/dense_3/kernel*
_output_shapes
:	?*
dtype0
?
 autoencoder/encoder/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" autoencoder/encoder/dense_3/bias
?
4autoencoder/encoder/dense_3/bias/Read/ReadVariableOpReadVariableOp autoencoder/encoder/dense_3/bias*
_output_shapes
:*
dtype0
?
"autoencoder/encoder/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"autoencoder/encoder/dense_4/kernel
?
6autoencoder/encoder/dense_4/kernel/Read/ReadVariableOpReadVariableOp"autoencoder/encoder/dense_4/kernel*
_output_shapes
:	?*
dtype0
?
 autoencoder/encoder/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" autoencoder/encoder/dense_4/bias
?
4autoencoder/encoder/dense_4/bias/Read/ReadVariableOpReadVariableOp autoencoder/encoder/dense_4/bias*
_output_shapes
:*
dtype0
?
"autoencoder/decoder/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"autoencoder/decoder/dense_5/kernel
?
6autoencoder/decoder/dense_5/kernel/Read/ReadVariableOpReadVariableOp"autoencoder/decoder/dense_5/kernel*
_output_shapes
:	?*
dtype0
?
 autoencoder/decoder/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" autoencoder/decoder/dense_5/bias
?
4autoencoder/decoder/dense_5/bias/Read/ReadVariableOpReadVariableOp autoencoder/decoder/dense_5/bias*
_output_shapes	
:?*
dtype0
?
-autoencoder/decoder/conv1d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*>
shared_name/-autoencoder/decoder/conv1d_transpose_2/kernel
?
Aautoencoder/decoder/conv1d_transpose_2/kernel/Read/ReadVariableOpReadVariableOp-autoencoder/decoder/conv1d_transpose_2/kernel*"
_output_shapes
:$*
dtype0
?
+autoencoder/decoder/conv1d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*<
shared_name-+autoencoder/decoder/conv1d_transpose_2/bias
?
?autoencoder/decoder/conv1d_transpose_2/bias/Read/ReadVariableOpReadVariableOp+autoencoder/decoder/conv1d_transpose_2/bias*
_output_shapes
:$*
dtype0
?
-autoencoder/decoder/conv1d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:7$*>
shared_name/-autoencoder/decoder/conv1d_transpose_3/kernel
?
Aautoencoder/decoder/conv1d_transpose_3/kernel/Read/ReadVariableOpReadVariableOp-autoencoder/decoder/conv1d_transpose_3/kernel*"
_output_shapes
:7$*
dtype0
?
+autoencoder/decoder/conv1d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:7*<
shared_name-+autoencoder/decoder/conv1d_transpose_3/bias
?
?autoencoder/decoder/conv1d_transpose_3/bias/Read/ReadVariableOpReadVariableOp+autoencoder/decoder/conv1d_transpose_3/bias*
_output_shapes
:7*
dtype0

NoOpNoOp
?/
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?.
value?.B?. B?.
|
encoder
decoder
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?
	conv0
		conv1

flat

dense_mean
dense_log_var
sampling
	variables
trainable_variables
regularization_losses
	keras_api
?
	dense
reshape
	conv1
	conv2
	variables
trainable_variables
regularization_losses
	keras_api
f
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
&12
'13
f
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
&12
'13
 
?
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
 
h

kernel
bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
h

kernel
bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
R
5	variables
6trainable_variables
7regularization_losses
8	keras_api
h

kernel
bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
h

 kernel
!bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
R
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
8
0
1
2
3
4
5
 6
!7
8
0
1
2
3
4
5
 6
!7
 
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
h

"kernel
#bias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
R
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
h

$kernel
%bias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
h

&kernel
'bias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
*
"0
#1
$2
%3
&4
'5
*
"0
#1
$2
%3
&4
'5
 
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
_]
VARIABLE_VALUE#autoencoder/encoder/conv1d_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!autoencoder/encoder/conv1d_2/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#autoencoder/encoder/conv1d_3/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!autoencoder/encoder/conv1d_3/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"autoencoder/encoder/dense_3/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE autoencoder/encoder/dense_3/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"autoencoder/encoder/dense_4/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE autoencoder/encoder/dense_4/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"autoencoder/decoder/dense_5/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE autoencoder/decoder/dense_5/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-autoencoder/decoder/conv1d_transpose_2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+autoencoder/decoder/conv1d_transpose_2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-autoencoder/decoder/conv1d_transpose_3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+autoencoder/decoder/conv1d_transpose_3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
 
 

0
1

0
1
 
?
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
-	variables
.trainable_variables
/regularization_losses

0
1

0
1
 
?
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
1	variables
2trainable_variables
3regularization_losses
 
 
 
?
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
5	variables
6trainable_variables
7regularization_losses

0
1

0
1
 
?
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
9	variables
:trainable_variables
;regularization_losses

 0
!1

 0
!1
 
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
=	variables
>trainable_variables
?regularization_losses
 
 
 
?
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
 
*
0
	1

2
3
4
5
 
 
 

"0
#1

"0
#1
 
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses

$0
%1

$0
%1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses

&0
'1

&0
'1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
 

0
1
2
3
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
 
 
 
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????d7*
dtype0* 
shape:?????????d7
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1#autoencoder/encoder/conv1d_2/kernel!autoencoder/encoder/conv1d_2/bias#autoencoder/encoder/conv1d_3/kernel!autoencoder/encoder/conv1d_3/bias"autoencoder/encoder/dense_3/kernel autoencoder/encoder/dense_3/bias"autoencoder/encoder/dense_4/kernel autoencoder/encoder/dense_4/bias"autoencoder/decoder/dense_5/kernel autoencoder/decoder/dense_5/bias-autoencoder/decoder/conv1d_transpose_2/kernel+autoencoder/decoder/conv1d_transpose_2/bias-autoencoder/decoder/conv1d_transpose_3/kernel+autoencoder/decoder/conv1d_transpose_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d7*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_207339
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename7autoencoder/encoder/conv1d_2/kernel/Read/ReadVariableOp5autoencoder/encoder/conv1d_2/bias/Read/ReadVariableOp7autoencoder/encoder/conv1d_3/kernel/Read/ReadVariableOp5autoencoder/encoder/conv1d_3/bias/Read/ReadVariableOp6autoencoder/encoder/dense_3/kernel/Read/ReadVariableOp4autoencoder/encoder/dense_3/bias/Read/ReadVariableOp6autoencoder/encoder/dense_4/kernel/Read/ReadVariableOp4autoencoder/encoder/dense_4/bias/Read/ReadVariableOp6autoencoder/decoder/dense_5/kernel/Read/ReadVariableOp4autoencoder/decoder/dense_5/bias/Read/ReadVariableOpAautoencoder/decoder/conv1d_transpose_2/kernel/Read/ReadVariableOp?autoencoder/decoder/conv1d_transpose_2/bias/Read/ReadVariableOpAautoencoder/decoder/conv1d_transpose_3/kernel/Read/ReadVariableOp?autoencoder/decoder/conv1d_transpose_3/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_207909
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename#autoencoder/encoder/conv1d_2/kernel!autoencoder/encoder/conv1d_2/bias#autoencoder/encoder/conv1d_3/kernel!autoencoder/encoder/conv1d_3/bias"autoencoder/encoder/dense_3/kernel autoencoder/encoder/dense_3/bias"autoencoder/encoder/dense_4/kernel autoencoder/encoder/dense_4/bias"autoencoder/decoder/dense_5/kernel autoencoder/decoder/dense_5/bias-autoencoder/decoder/conv1d_transpose_2/kernel+autoencoder/decoder/conv1d_transpose_2/bias-autoencoder/decoder/conv1d_transpose_3/kernel+autoencoder/decoder/conv1d_transpose_3/bias*
Tin
2*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_207961??

?
?
C__inference_decoder_layer_call_and_return_conditional_losses_207121

inputs9
&dense_5_matmul_readvariableop_resource:	?6
'dense_5_biasadd_readvariableop_resource:	?^
Hconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource:$@
2conv1d_transpose_2_biasadd_readvariableop_resource:$^
Hconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource:7$@
2conv1d_transpose_3_biasadd_readvariableop_resource:7
identity??)conv1d_transpose_2/BiasAdd/ReadVariableOp??conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?)conv1d_transpose_3/BiasAdd/ReadVariableOp??conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0z
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????Y
reshape_1/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
reshape_1/ReshapeReshapedense_5/Relu:activations:0 reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????b
conv1d_transpose_2/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:p
&conv1d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv1d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv1d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv1d_transpose_2/strided_sliceStridedSlice!conv1d_transpose_2/Shape:output:0/conv1d_transpose_2/strided_slice/stack:output:01conv1d_transpose_2/strided_slice/stack_1:output:01conv1d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
(conv1d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*conv1d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv1d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv1d_transpose_2/strided_slice_1StridedSlice!conv1d_transpose_2/Shape:output:01conv1d_transpose_2/strided_slice_1/stack:output:03conv1d_transpose_2/strided_slice_1/stack_1:output:03conv1d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv1d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv1d_transpose_2/mulMul+conv1d_transpose_2/strided_slice_1:output:0!conv1d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: Z
conv1d_transpose_2/add/yConst*
_output_shapes
: *
dtype0*
value	B : 
conv1d_transpose_2/addAddV2conv1d_transpose_2/mul:z:0!conv1d_transpose_2/add/y:output:0*
T0*
_output_shapes
: \
conv1d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :$?
conv1d_transpose_2/stackPack)conv1d_transpose_2/strided_slice:output:0conv1d_transpose_2/add:z:0#conv1d_transpose_2/stack/2:output:0*
N*
T0*
_output_shapes
:t
2conv1d_transpose_2/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
.conv1d_transpose_2/conv1d_transpose/ExpandDims
ExpandDimsreshape_1/Reshape:output:0;conv1d_transpose_2/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
dtype0v
4conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
0conv1d_transpose_2/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:$?
7conv1d_transpose_2/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1conv1d_transpose_2/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_2/stack:output:0@conv1d_transpose_2/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
9conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3conv1d_transpose_2/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_2/stack:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask}
3conv1d_transpose_2/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:q
/conv1d_transpose_2/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*conv1d_transpose_2/conv1d_transpose/concatConcatV2:conv1d_transpose_2/conv1d_transpose/strided_slice:output:0<conv1d_transpose_2/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_2/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_2/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
#conv1d_transpose_2/conv1d_transposeConv2DBackpropInput3conv1d_transpose_2/conv1d_transpose/concat:output:09conv1d_transpose_2/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_2/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????2$*
paddingVALID*
strides
?
+conv1d_transpose_2/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_2/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims
?
)conv1d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
conv1d_transpose_2/BiasAddBiasAdd4conv1d_transpose_2/conv1d_transpose/Squeeze:output:01conv1d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$z
conv1d_transpose_2/ReluRelu#conv1d_transpose_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$m
conv1d_transpose_3/ShapeShape%conv1d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:p
&conv1d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv1d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv1d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv1d_transpose_3/strided_sliceStridedSlice!conv1d_transpose_3/Shape:output:0/conv1d_transpose_3/strided_slice/stack:output:01conv1d_transpose_3/strided_slice/stack_1:output:01conv1d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
(conv1d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*conv1d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv1d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv1d_transpose_3/strided_slice_1StridedSlice!conv1d_transpose_3/Shape:output:01conv1d_transpose_3/strided_slice_1/stack:output:03conv1d_transpose_3/strided_slice_1/stack_1:output:03conv1d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv1d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv1d_transpose_3/mulMul+conv1d_transpose_3/strided_slice_1:output:0!conv1d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: Z
conv1d_transpose_3/add/yConst*
_output_shapes
: *
dtype0*
value	B : 
conv1d_transpose_3/addAddV2conv1d_transpose_3/mul:z:0!conv1d_transpose_3/add/y:output:0*
T0*
_output_shapes
: \
conv1d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :7?
conv1d_transpose_3/stackPack)conv1d_transpose_3/strided_slice:output:0conv1d_transpose_3/add:z:0#conv1d_transpose_3/stack/2:output:0*
N*
T0*
_output_shapes
:t
2conv1d_transpose_3/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
.conv1d_transpose_3/conv1d_transpose/ExpandDims
ExpandDims%conv1d_transpose_2/Relu:activations:0;conv1d_transpose_3/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2$?
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
dtype0v
4conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
0conv1d_transpose_3/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:7$?
7conv1d_transpose_3/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1conv1d_transpose_3/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_3/stack:output:0@conv1d_transpose_3/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
9conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3conv1d_transpose_3/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_3/stack:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask}
3conv1d_transpose_3/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:q
/conv1d_transpose_3/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*conv1d_transpose_3/conv1d_transpose/concatConcatV2:conv1d_transpose_3/conv1d_transpose/strided_slice:output:0<conv1d_transpose_3/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_3/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_3/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
#conv1d_transpose_3/conv1d_transposeConv2DBackpropInput3conv1d_transpose_3/conv1d_transpose/concat:output:09conv1d_transpose_3/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_3/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d7*
paddingVALID*
strides
?
+conv1d_transpose_3/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_3/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????d7*
squeeze_dims
?
)conv1d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0?
conv1d_transpose_3/BiasAddBiasAdd4conv1d_transpose_3/conv1d_transpose/Squeeze:output:01conv1d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d7z
conv1d_transpose_3/ReluRelu#conv1d_transpose_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d7x
IdentityIdentity%conv1d_transpose_3/Relu:activations:0^NoOp*
T0*+
_output_shapes
:?????????d7?
NoOpNoOp*^conv1d_transpose_2/BiasAdd/ReadVariableOp@^conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp*^conv1d_transpose_3/BiasAdd/ReadVariableOp@^conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2V
)conv1d_transpose_2/BiasAdd/ReadVariableOp)conv1d_transpose_2/BiasAdd/ReadVariableOp2?
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2V
)conv1d_transpose_3/BiasAdd/ReadVariableOp)conv1d_transpose_3/BiasAdd/ReadVariableOp2?
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?)
?
__inference__traced_save_207909
file_prefixB
>savev2_autoencoder_encoder_conv1d_2_kernel_read_readvariableop@
<savev2_autoencoder_encoder_conv1d_2_bias_read_readvariableopB
>savev2_autoencoder_encoder_conv1d_3_kernel_read_readvariableop@
<savev2_autoencoder_encoder_conv1d_3_bias_read_readvariableopA
=savev2_autoencoder_encoder_dense_3_kernel_read_readvariableop?
;savev2_autoencoder_encoder_dense_3_bias_read_readvariableopA
=savev2_autoencoder_encoder_dense_4_kernel_read_readvariableop?
;savev2_autoencoder_encoder_dense_4_bias_read_readvariableopA
=savev2_autoencoder_decoder_dense_5_kernel_read_readvariableop?
;savev2_autoencoder_decoder_dense_5_bias_read_readvariableopL
Hsavev2_autoencoder_decoder_conv1d_transpose_2_kernel_read_readvariableopJ
Fsavev2_autoencoder_decoder_conv1d_transpose_2_bias_read_readvariableopL
Hsavev2_autoencoder_decoder_conv1d_transpose_3_kernel_read_readvariableopJ
Fsavev2_autoencoder_decoder_conv1d_transpose_3_bias_read_readvariableop
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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0>savev2_autoencoder_encoder_conv1d_2_kernel_read_readvariableop<savev2_autoencoder_encoder_conv1d_2_bias_read_readvariableop>savev2_autoencoder_encoder_conv1d_3_kernel_read_readvariableop<savev2_autoencoder_encoder_conv1d_3_bias_read_readvariableop=savev2_autoencoder_encoder_dense_3_kernel_read_readvariableop;savev2_autoencoder_encoder_dense_3_bias_read_readvariableop=savev2_autoencoder_encoder_dense_4_kernel_read_readvariableop;savev2_autoencoder_encoder_dense_4_bias_read_readvariableop=savev2_autoencoder_decoder_dense_5_kernel_read_readvariableop;savev2_autoencoder_decoder_dense_5_bias_read_readvariableopHsavev2_autoencoder_decoder_conv1d_transpose_2_kernel_read_readvariableopFsavev2_autoencoder_decoder_conv1d_transpose_2_bias_read_readvariableopHsavev2_autoencoder_decoder_conv1d_transpose_3_kernel_read_readvariableopFsavev2_autoencoder_decoder_conv1d_transpose_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :7$:$:$::	?::	?::	?:?:$:$:7$:7: 2(
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
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%	!

_output_shapes
:	?:!


_output_shapes	
:?:($
"
_output_shapes
:$: 

_output_shapes
:$:($
"
_output_shapes
:7$: 

_output_shapes
:7:

_output_shapes
: 
?,
?
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_206873

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
?
?
G__inference_autoencoder_layer_call_and_return_conditional_losses_207147

inputs$
encoder_207006:7$
encoder_207008:$$
encoder_207010:$
encoder_207012:!
encoder_207014:	?
encoder_207016:!
encoder_207018:	?
encoder_207020:!
decoder_207122:	?
decoder_207124:	?$
decoder_207126:$
decoder_207128:$$
decoder_207130:7$
decoder_207132:7
identity

identity_1??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_207006encoder_207008encoder_207010encoder_207012encoder_207014encoder_207016encoder_207018encoder_207020*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_207005?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:2decoder_207122decoder_207124decoder_207126decoder_207128decoder_207130decoder_207132*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d7*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_207121l
SquareSquare(encoder/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????r
subSub(encoder/StatefulPartitionedCall:output:1
Square:y:0*
T0*'
_output_shapes
:?????????f
ExpExp(encoder/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:?????????P
sub_1Subsub:z:0Exp:y:0*
T0*'
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Y
addAddV2	sub_1:z:0add/y:output:0*
T0*'
_output_shapes
:?????????V
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
:?????????d7G

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
3:?????????d7: : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:S O
+
_output_shapes
:?????????d7
 
_user_specified_nameinputs
?
?
(__inference_encoder_layer_call_fn_207564

inputs
unknown:7$
	unknown_0:$
	unknown_1:$
	unknown_2:
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
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
9:?????????:?????????:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_207005o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d7: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????d7
 
_user_specified_nameinputs
?
?
,__inference_autoencoder_layer_call_fn_207179
input_1
unknown:7$
	unknown_0:$
	unknown_1:$
	unknown_2:
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
	unknown_7:	?
	unknown_8:	?
	unknown_9:$

unknown_10:$ 

unknown_11:7$

unknown_12:7
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
:?????????d7: *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_autoencoder_layer_call_and_return_conditional_losses_207147s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????d7: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????d7
!
_user_specified_name	input_1
?
?
$__inference_signature_wrapper_207339
input_1
unknown:7$
	unknown_0:$
	unknown_1:$
	unknown_2:
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
	unknown_7:	?
	unknown_8:	?
	unknown_9:$

unknown_10:$ 

unknown_11:7$

unknown_12:7
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
:?????????d7*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_206827s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????d7: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????d7
!
_user_specified_name	input_1
?
?
3__inference_conv1d_transpose_3_layer_call_fn_207802

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
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_206926|
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
?,
?
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_207793

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
?K
?
C__inference_encoder_layer_call_and_return_conditional_losses_207005

inputsJ
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:7$6
(conv1d_2_biasadd_readvariableop_resource:$J
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:$6
(conv1d_3_biasadd_readvariableop_resource:9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:9
&dense_4_matmul_readvariableop_resource:	?5
'dense_4_biasadd_readvariableop_resource:
identity

identity_1

identity_2??conv1d_2/BiasAdd/ReadVariableOp?+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp?conv1d_3/BiasAdd/ReadVariableOp?+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOpi
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d_2/Conv1D/ExpandDims
ExpandDimsinputs'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d7?
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:7$?
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????2$*
paddingVALID*
strides
?
conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims

??????????
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$f
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$i
conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d_3/Conv1D/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0'conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2$?
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
dtype0b
 conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_3/Conv1D/ExpandDims_1
ExpandDims3conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:$?
conv1d_3/Conv1DConv2D#conv1d_3/Conv1D/ExpandDims:output:0%conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv1d_3/Conv1D/SqueezeSqueezeconv1d_3/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1d_3/BiasAddBiasAdd conv1d_3/Conv1D/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????f
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten_1/ReshapeReshapeconv1d_3/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:???????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_4/MatMulMatMulflatten_1/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X
sampling_1/ShapeShapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:h
sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sampling_1/strided_sliceStridedSlicesampling_1/Shape:output:0'sampling_1/strided_slice/stack:output:0)sampling_1/strided_slice/stack_1:output:0)sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
sampling_1/Shape_1Shapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:j
 sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:l
"sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sampling_1/strided_slice_1StridedSlicesampling_1/Shape_1:output:0)sampling_1/strided_slice_1/stack:output:0+sampling_1/strided_slice_1/stack_1:output:0+sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
sampling_1/random_normal/shapePack!sampling_1/strided_slice:output:0#sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:b
sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    d
sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
-sampling_1/random_normal/RandomStandardNormalRandomStandardNormal'sampling_1/random_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2????
sampling_1/random_normal/mulMul6sampling_1/random_normal/RandomStandardNormal:output:0(sampling_1/random_normal/stddev:output:0*
T0*'
_output_shapes
:??????????
sampling_1/random_normalAddV2 sampling_1/random_normal/mul:z:0&sampling_1/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????U
sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?|
sampling_1/mulMulsampling_1/mul/x:output:0dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????[
sampling_1/ExpExpsampling_1/mul:z:0*
T0*'
_output_shapes
:?????????{
sampling_1/mul_1Mulsampling_1/Exp:y:0sampling_1/random_normal:z:0*
T0*'
_output_shapes
:?????????y
sampling_1/addAddV2dense_3/BiasAdd:output:0sampling_1/mul_1:z:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????i

Identity_1Identitydense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????c

Identity_2Identitysampling_1/add:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d7: : : : : : : : 2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????d7
 
_user_specified_nameinputs
?,
?
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_206926

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
?,
?
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_207844

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
?
?
G__inference_autoencoder_layer_call_and_return_conditional_losses_207304
input_1$
encoder_207260:7$
encoder_207262:$$
encoder_207264:$
encoder_207266:!
encoder_207268:	?
encoder_207270:!
encoder_207272:	?
encoder_207274:!
decoder_207279:	?
decoder_207281:	?$
decoder_207283:$
decoder_207285:$$
decoder_207287:7$
decoder_207289:7
identity

identity_1??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_207260encoder_207262encoder_207264encoder_207266encoder_207268encoder_207270encoder_207272encoder_207274*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_207005?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:2decoder_207279decoder_207281decoder_207283decoder_207285decoder_207287decoder_207289*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d7*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_207121l
SquareSquare(encoder/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????r
subSub(encoder/StatefulPartitionedCall:output:1
Square:y:0*
T0*'
_output_shapes
:?????????f
ExpExp(encoder/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:?????????P
sub_1Subsub:z:0Exp:y:0*
T0*'
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Y
addAddV2	sub_1:z:0add/y:output:0*
T0*'
_output_shapes
:?????????V
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
:?????????d7G

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
3:?????????d7: : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:T P
+
_output_shapes
:?????????d7
!
_user_specified_name	input_1
?
?
3__inference_conv1d_transpose_2_layer_call_fn_207751

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
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_206873|
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
?
,__inference_autoencoder_layer_call_fn_207373

inputs
unknown:7$
	unknown_0:$
	unknown_1:$
	unknown_2:
	unknown_3:	?
	unknown_4:
	unknown_5:	?
	unknown_6:
	unknown_7:	?
	unknown_8:	?
	unknown_9:$

unknown_10:$ 

unknown_11:7$

unknown_12:7
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
:?????????d7: *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_autoencoder_layer_call_and_return_conditional_losses_207147s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????d7: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????d7
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_206827
input_1^
Hautoencoder_encoder_conv1d_2_conv1d_expanddims_1_readvariableop_resource:7$J
<autoencoder_encoder_conv1d_2_biasadd_readvariableop_resource:$^
Hautoencoder_encoder_conv1d_3_conv1d_expanddims_1_readvariableop_resource:$J
<autoencoder_encoder_conv1d_3_biasadd_readvariableop_resource:M
:autoencoder_encoder_dense_3_matmul_readvariableop_resource:	?I
;autoencoder_encoder_dense_3_biasadd_readvariableop_resource:M
:autoencoder_encoder_dense_4_matmul_readvariableop_resource:	?I
;autoencoder_encoder_dense_4_biasadd_readvariableop_resource:M
:autoencoder_decoder_dense_5_matmul_readvariableop_resource:	?J
;autoencoder_decoder_dense_5_biasadd_readvariableop_resource:	?r
\autoencoder_decoder_conv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource:$T
Fautoencoder_decoder_conv1d_transpose_2_biasadd_readvariableop_resource:$r
\autoencoder_decoder_conv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource:7$T
Fautoencoder_decoder_conv1d_transpose_3_biasadd_readvariableop_resource:7
identity??=autoencoder/decoder/conv1d_transpose_2/BiasAdd/ReadVariableOp?Sautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?=autoencoder/decoder/conv1d_transpose_3/BiasAdd/ReadVariableOp?Sautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?2autoencoder/decoder/dense_5/BiasAdd/ReadVariableOp?1autoencoder/decoder/dense_5/MatMul/ReadVariableOp?3autoencoder/encoder/conv1d_2/BiasAdd/ReadVariableOp??autoencoder/encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp?3autoencoder/encoder/conv1d_3/BiasAdd/ReadVariableOp??autoencoder/encoder/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp?2autoencoder/encoder/dense_3/BiasAdd/ReadVariableOp?1autoencoder/encoder/dense_3/MatMul/ReadVariableOp?2autoencoder/encoder/dense_4/BiasAdd/ReadVariableOp?1autoencoder/encoder/dense_4/MatMul/ReadVariableOp}
2autoencoder/encoder/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.autoencoder/encoder/conv1d_2/Conv1D/ExpandDims
ExpandDimsinput_1;autoencoder/encoder/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d7?
?autoencoder/encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpHautoencoder_encoder_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
dtype0v
4autoencoder/encoder/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
0autoencoder/encoder/conv1d_2/Conv1D/ExpandDims_1
ExpandDimsGautoencoder/encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0=autoencoder/encoder/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:7$?
#autoencoder/encoder/conv1d_2/Conv1DConv2D7autoencoder/encoder/conv1d_2/Conv1D/ExpandDims:output:09autoencoder/encoder/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????2$*
paddingVALID*
strides
?
+autoencoder/encoder/conv1d_2/Conv1D/SqueezeSqueeze,autoencoder/encoder/conv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims

??????????
3autoencoder/encoder/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_encoder_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
$autoencoder/encoder/conv1d_2/BiasAddBiasAdd4autoencoder/encoder/conv1d_2/Conv1D/Squeeze:output:0;autoencoder/encoder/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$?
!autoencoder/encoder/conv1d_2/ReluRelu-autoencoder/encoder/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$}
2autoencoder/encoder/conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.autoencoder/encoder/conv1d_3/Conv1D/ExpandDims
ExpandDims/autoencoder/encoder/conv1d_2/Relu:activations:0;autoencoder/encoder/conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2$?
?autoencoder/encoder/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpHautoencoder_encoder_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
dtype0v
4autoencoder/encoder/conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
0autoencoder/encoder/conv1d_3/Conv1D/ExpandDims_1
ExpandDimsGautoencoder/encoder/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0=autoencoder/encoder/conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:$?
#autoencoder/encoder/conv1d_3/Conv1DConv2D7autoencoder/encoder/conv1d_3/Conv1D/ExpandDims:output:09autoencoder/encoder/conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
+autoencoder/encoder/conv1d_3/Conv1D/SqueezeSqueeze,autoencoder/encoder/conv1d_3/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
3autoencoder/encoder/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_encoder_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
$autoencoder/encoder/conv1d_3/BiasAddBiasAdd4autoencoder/encoder/conv1d_3/Conv1D/Squeeze:output:0;autoencoder/encoder/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:??????????
!autoencoder/encoder/conv1d_3/ReluRelu-autoencoder/encoder/conv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????t
#autoencoder/encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
%autoencoder/encoder/flatten_1/ReshapeReshape/autoencoder/encoder/conv1d_3/Relu:activations:0,autoencoder/encoder/flatten_1/Const:output:0*
T0*(
_output_shapes
:???????????
1autoencoder/encoder/dense_3/MatMul/ReadVariableOpReadVariableOp:autoencoder_encoder_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
"autoencoder/encoder/dense_3/MatMulMatMul.autoencoder/encoder/flatten_1/Reshape:output:09autoencoder/encoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
2autoencoder/encoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_encoder_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
#autoencoder/encoder/dense_3/BiasAddBiasAdd,autoencoder/encoder/dense_3/MatMul:product:0:autoencoder/encoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
1autoencoder/encoder/dense_4/MatMul/ReadVariableOpReadVariableOp:autoencoder_encoder_dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
"autoencoder/encoder/dense_4/MatMulMatMul.autoencoder/encoder/flatten_1/Reshape:output:09autoencoder/encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
2autoencoder/encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
#autoencoder/encoder/dense_4/BiasAddBiasAdd,autoencoder/encoder/dense_4/MatMul:product:0:autoencoder/encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$autoencoder/encoder/sampling_1/ShapeShape,autoencoder/encoder/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:|
2autoencoder/encoder/sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4autoencoder/encoder/sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4autoencoder/encoder/sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,autoencoder/encoder/sampling_1/strided_sliceStridedSlice-autoencoder/encoder/sampling_1/Shape:output:0;autoencoder/encoder/sampling_1/strided_slice/stack:output:0=autoencoder/encoder/sampling_1/strided_slice/stack_1:output:0=autoencoder/encoder/sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
&autoencoder/encoder/sampling_1/Shape_1Shape,autoencoder/encoder/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:~
4autoencoder/encoder/sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
6autoencoder/encoder/sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6autoencoder/encoder/sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.autoencoder/encoder/sampling_1/strided_slice_1StridedSlice/autoencoder/encoder/sampling_1/Shape_1:output:0=autoencoder/encoder/sampling_1/strided_slice_1/stack:output:0?autoencoder/encoder/sampling_1/strided_slice_1/stack_1:output:0?autoencoder/encoder/sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2autoencoder/encoder/sampling_1/random_normal/shapePack5autoencoder/encoder/sampling_1/strided_slice:output:07autoencoder/encoder/sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:v
1autoencoder/encoder/sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    x
3autoencoder/encoder/sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Aautoencoder/encoder/sampling_1/random_normal/RandomStandardNormalRandomStandardNormal;autoencoder/encoder/sampling_1/random_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2????
0autoencoder/encoder/sampling_1/random_normal/mulMulJautoencoder/encoder/sampling_1/random_normal/RandomStandardNormal:output:0<autoencoder/encoder/sampling_1/random_normal/stddev:output:0*
T0*'
_output_shapes
:??????????
,autoencoder/encoder/sampling_1/random_normalAddV24autoencoder/encoder/sampling_1/random_normal/mul:z:0:autoencoder/encoder/sampling_1/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????i
$autoencoder/encoder/sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
"autoencoder/encoder/sampling_1/mulMul-autoencoder/encoder/sampling_1/mul/x:output:0,autoencoder/encoder/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
"autoencoder/encoder/sampling_1/ExpExp&autoencoder/encoder/sampling_1/mul:z:0*
T0*'
_output_shapes
:??????????
$autoencoder/encoder/sampling_1/mul_1Mul&autoencoder/encoder/sampling_1/Exp:y:00autoencoder/encoder/sampling_1/random_normal:z:0*
T0*'
_output_shapes
:??????????
"autoencoder/encoder/sampling_1/addAddV2,autoencoder/encoder/dense_3/BiasAdd:output:0(autoencoder/encoder/sampling_1/mul_1:z:0*
T0*'
_output_shapes
:??????????
1autoencoder/decoder/dense_5/MatMul/ReadVariableOpReadVariableOp:autoencoder_decoder_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
"autoencoder/decoder/dense_5/MatMulMatMul&autoencoder/encoder/sampling_1/add:z:09autoencoder/decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
2autoencoder/decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
#autoencoder/decoder/dense_5/BiasAddBiasAdd,autoencoder/decoder/dense_5/MatMul:product:0:autoencoder/decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
 autoencoder/decoder/dense_5/ReluRelu,autoencoder/decoder/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
#autoencoder/decoder/reshape_1/ShapeShape.autoencoder/decoder/dense_5/Relu:activations:0*
T0*
_output_shapes
:{
1autoencoder/decoder/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3autoencoder/decoder/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3autoencoder/decoder/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+autoencoder/decoder/reshape_1/strided_sliceStridedSlice,autoencoder/decoder/reshape_1/Shape:output:0:autoencoder/decoder/reshape_1/strided_slice/stack:output:0<autoencoder/decoder/reshape_1/strided_slice/stack_1:output:0<autoencoder/decoder/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-autoencoder/decoder/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-autoencoder/decoder/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
+autoencoder/decoder/reshape_1/Reshape/shapePack4autoencoder/decoder/reshape_1/strided_slice:output:06autoencoder/decoder/reshape_1/Reshape/shape/1:output:06autoencoder/decoder/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
%autoencoder/decoder/reshape_1/ReshapeReshape.autoencoder/decoder/dense_5/Relu:activations:04autoencoder/decoder/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:??????????
,autoencoder/decoder/conv1d_transpose_2/ShapeShape.autoencoder/decoder/reshape_1/Reshape:output:0*
T0*
_output_shapes
:?
:autoencoder/decoder/conv1d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<autoencoder/decoder/conv1d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<autoencoder/decoder/conv1d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4autoencoder/decoder/conv1d_transpose_2/strided_sliceStridedSlice5autoencoder/decoder/conv1d_transpose_2/Shape:output:0Cautoencoder/decoder/conv1d_transpose_2/strided_slice/stack:output:0Eautoencoder/decoder/conv1d_transpose_2/strided_slice/stack_1:output:0Eautoencoder/decoder/conv1d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
<autoencoder/decoder/conv1d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
>autoencoder/decoder/conv1d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>autoencoder/decoder/conv1d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6autoencoder/decoder/conv1d_transpose_2/strided_slice_1StridedSlice5autoencoder/decoder/conv1d_transpose_2/Shape:output:0Eautoencoder/decoder/conv1d_transpose_2/strided_slice_1/stack:output:0Gautoencoder/decoder/conv1d_transpose_2/strided_slice_1/stack_1:output:0Gautoencoder/decoder/conv1d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,autoencoder/decoder/conv1d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
*autoencoder/decoder/conv1d_transpose_2/mulMul?autoencoder/decoder/conv1d_transpose_2/strided_slice_1:output:05autoencoder/decoder/conv1d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: n
,autoencoder/decoder/conv1d_transpose_2/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
*autoencoder/decoder/conv1d_transpose_2/addAddV2.autoencoder/decoder/conv1d_transpose_2/mul:z:05autoencoder/decoder/conv1d_transpose_2/add/y:output:0*
T0*
_output_shapes
: p
.autoencoder/decoder/conv1d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :$?
,autoencoder/decoder/conv1d_transpose_2/stackPack=autoencoder/decoder/conv1d_transpose_2/strided_slice:output:0.autoencoder/decoder/conv1d_transpose_2/add:z:07autoencoder/decoder/conv1d_transpose_2/stack/2:output:0*
N*
T0*
_output_shapes
:?
Fautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
Bautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims
ExpandDims.autoencoder/decoder/reshape_1/Reshape:output:0Oautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
Sautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp\autoencoder_decoder_conv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
dtype0?
Hautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Dautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1
ExpandDims[autoencoder/decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Qautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:$?
Kautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Mautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Mautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Eautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/strided_sliceStridedSlice5autoencoder/decoder/conv1d_transpose_2/stack:output:0Tautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack:output:0Vautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1:output:0Vautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Mautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Oautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Oautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Gautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1StridedSlice5autoencoder/decoder/conv1d_transpose_2/stack:output:0Vautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack:output:0Xautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1:output:0Xautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Gautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>autoencoder/decoder/conv1d_transpose_2/conv1d_transpose/concatConcatV2Nautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/strided_slice:output:0Pautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/concat/values_1:output:0Pautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1:output:0Lautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
7autoencoder/decoder/conv1d_transpose_2/conv1d_transposeConv2DBackpropInputGautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/concat:output:0Mautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1:output:0Kautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????2$*
paddingVALID*
strides
?
?autoencoder/decoder/conv1d_transpose_2/conv1d_transpose/SqueezeSqueeze@autoencoder/decoder/conv1d_transpose_2/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims
?
=autoencoder/decoder/conv1d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpFautoencoder_decoder_conv1d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
.autoencoder/decoder/conv1d_transpose_2/BiasAddBiasAddHautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/Squeeze:output:0Eautoencoder/decoder/conv1d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$?
+autoencoder/decoder/conv1d_transpose_2/ReluRelu7autoencoder/decoder/conv1d_transpose_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$?
,autoencoder/decoder/conv1d_transpose_3/ShapeShape9autoencoder/decoder/conv1d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:?
:autoencoder/decoder/conv1d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<autoencoder/decoder/conv1d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<autoencoder/decoder/conv1d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4autoencoder/decoder/conv1d_transpose_3/strided_sliceStridedSlice5autoencoder/decoder/conv1d_transpose_3/Shape:output:0Cautoencoder/decoder/conv1d_transpose_3/strided_slice/stack:output:0Eautoencoder/decoder/conv1d_transpose_3/strided_slice/stack_1:output:0Eautoencoder/decoder/conv1d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
<autoencoder/decoder/conv1d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
>autoencoder/decoder/conv1d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>autoencoder/decoder/conv1d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6autoencoder/decoder/conv1d_transpose_3/strided_slice_1StridedSlice5autoencoder/decoder/conv1d_transpose_3/Shape:output:0Eautoencoder/decoder/conv1d_transpose_3/strided_slice_1/stack:output:0Gautoencoder/decoder/conv1d_transpose_3/strided_slice_1/stack_1:output:0Gautoencoder/decoder/conv1d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,autoencoder/decoder/conv1d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
*autoencoder/decoder/conv1d_transpose_3/mulMul?autoencoder/decoder/conv1d_transpose_3/strided_slice_1:output:05autoencoder/decoder/conv1d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: n
,autoencoder/decoder/conv1d_transpose_3/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
*autoencoder/decoder/conv1d_transpose_3/addAddV2.autoencoder/decoder/conv1d_transpose_3/mul:z:05autoencoder/decoder/conv1d_transpose_3/add/y:output:0*
T0*
_output_shapes
: p
.autoencoder/decoder/conv1d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :7?
,autoencoder/decoder/conv1d_transpose_3/stackPack=autoencoder/decoder/conv1d_transpose_3/strided_slice:output:0.autoencoder/decoder/conv1d_transpose_3/add:z:07autoencoder/decoder/conv1d_transpose_3/stack/2:output:0*
N*
T0*
_output_shapes
:?
Fautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
Bautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims
ExpandDims9autoencoder/decoder/conv1d_transpose_2/Relu:activations:0Oautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2$?
Sautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp\autoencoder_decoder_conv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
dtype0?
Hautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Dautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1
ExpandDims[autoencoder/decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Qautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:7$?
Kautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Mautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Mautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Eautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/strided_sliceStridedSlice5autoencoder/decoder/conv1d_transpose_3/stack:output:0Tautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack:output:0Vautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1:output:0Vautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Mautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Oautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Oautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Gautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1StridedSlice5autoencoder/decoder/conv1d_transpose_3/stack:output:0Vautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack:output:0Xautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1:output:0Xautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
Gautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:?
Cautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>autoencoder/decoder/conv1d_transpose_3/conv1d_transpose/concatConcatV2Nautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/strided_slice:output:0Pautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/concat/values_1:output:0Pautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1:output:0Lautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
7autoencoder/decoder/conv1d_transpose_3/conv1d_transposeConv2DBackpropInputGautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/concat:output:0Mautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1:output:0Kautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d7*
paddingVALID*
strides
?
?autoencoder/decoder/conv1d_transpose_3/conv1d_transpose/SqueezeSqueeze@autoencoder/decoder/conv1d_transpose_3/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????d7*
squeeze_dims
?
=autoencoder/decoder/conv1d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpFautoencoder_decoder_conv1d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0?
.autoencoder/decoder/conv1d_transpose_3/BiasAddBiasAddHautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/Squeeze:output:0Eautoencoder/decoder/conv1d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d7?
+autoencoder/decoder/conv1d_transpose_3/ReluRelu7autoencoder/decoder/conv1d_transpose_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d7|
autoencoder/SquareSquare,autoencoder/encoder/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
autoencoder/subSub,autoencoder/encoder/dense_4/BiasAdd:output:0autoencoder/Square:y:0*
T0*'
_output_shapes
:?????????v
autoencoder/ExpExp,autoencoder/encoder/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????t
autoencoder/sub_1Subautoencoder/sub:z:0autoencoder/Exp:y:0*
T0*'
_output_shapes
:?????????V
autoencoder/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??}
autoencoder/addAddV2autoencoder/sub_1:z:0autoencoder/add/y:output:0*
T0*'
_output_shapes
:?????????b
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
IdentityIdentity9autoencoder/decoder/conv1d_transpose_3/Relu:activations:0^NoOp*
T0*+
_output_shapes
:?????????d7?
NoOpNoOp>^autoencoder/decoder/conv1d_transpose_2/BiasAdd/ReadVariableOpT^autoencoder/decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp>^autoencoder/decoder/conv1d_transpose_3/BiasAdd/ReadVariableOpT^autoencoder/decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp3^autoencoder/decoder/dense_5/BiasAdd/ReadVariableOp2^autoencoder/decoder/dense_5/MatMul/ReadVariableOp4^autoencoder/encoder/conv1d_2/BiasAdd/ReadVariableOp@^autoencoder/encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp4^autoencoder/encoder/conv1d_3/BiasAdd/ReadVariableOp@^autoencoder/encoder/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp3^autoencoder/encoder/dense_3/BiasAdd/ReadVariableOp2^autoencoder/encoder/dense_3/MatMul/ReadVariableOp3^autoencoder/encoder/dense_4/BiasAdd/ReadVariableOp2^autoencoder/encoder/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????d7: : : : : : : : : : : : : : 2~
=autoencoder/decoder/conv1d_transpose_2/BiasAdd/ReadVariableOp=autoencoder/decoder/conv1d_transpose_2/BiasAdd/ReadVariableOp2?
Sautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpSautoencoder/decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2~
=autoencoder/decoder/conv1d_transpose_3/BiasAdd/ReadVariableOp=autoencoder/decoder/conv1d_transpose_3/BiasAdd/ReadVariableOp2?
Sautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpSautoencoder/decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp2h
2autoencoder/decoder/dense_5/BiasAdd/ReadVariableOp2autoencoder/decoder/dense_5/BiasAdd/ReadVariableOp2f
1autoencoder/decoder/dense_5/MatMul/ReadVariableOp1autoencoder/decoder/dense_5/MatMul/ReadVariableOp2j
3autoencoder/encoder/conv1d_2/BiasAdd/ReadVariableOp3autoencoder/encoder/conv1d_2/BiasAdd/ReadVariableOp2?
?autoencoder/encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp?autoencoder/encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2j
3autoencoder/encoder/conv1d_3/BiasAdd/ReadVariableOp3autoencoder/encoder/conv1d_3/BiasAdd/ReadVariableOp2?
?autoencoder/encoder/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp?autoencoder/encoder/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2h
2autoencoder/encoder/dense_3/BiasAdd/ReadVariableOp2autoencoder/encoder/dense_3/BiasAdd/ReadVariableOp2f
1autoencoder/encoder/dense_3/MatMul/ReadVariableOp1autoencoder/encoder/dense_3/MatMul/ReadVariableOp2h
2autoencoder/encoder/dense_4/BiasAdd/ReadVariableOp2autoencoder/encoder/dense_4/BiasAdd/ReadVariableOp2f
1autoencoder/encoder/dense_4/MatMul/ReadVariableOp1autoencoder/encoder/dense_4/MatMul/ReadVariableOp:T P
+
_output_shapes
:?????????d7
!
_user_specified_name	input_1
?K
?
C__inference_encoder_layer_call_and_return_conditional_losses_207629

inputsJ
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:7$6
(conv1d_2_biasadd_readvariableop_resource:$J
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:$6
(conv1d_3_biasadd_readvariableop_resource:9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:9
&dense_4_matmul_readvariableop_resource:	?5
'dense_4_biasadd_readvariableop_resource:
identity

identity_1

identity_2??conv1d_2/BiasAdd/ReadVariableOp?+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp?conv1d_3/BiasAdd/ReadVariableOp?+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOpi
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d_2/Conv1D/ExpandDims
ExpandDimsinputs'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d7?
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:7$?
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????2$*
paddingVALID*
strides
?
conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims

??????????
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$f
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$i
conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d_3/Conv1D/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0'conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2$?
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
dtype0b
 conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d_3/Conv1D/ExpandDims_1
ExpandDims3conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:$?
conv1d_3/Conv1DConv2D#conv1d_3/Conv1D/ExpandDims:output:0%conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv1d_3/Conv1D/SqueezeSqueezeconv1d_3/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv1d_3/BiasAddBiasAdd conv1d_3/Conv1D/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????f
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten_1/ReshapeReshapeconv1d_3/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:???????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_4/MatMulMatMulflatten_1/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X
sampling_1/ShapeShapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:h
sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sampling_1/strided_sliceStridedSlicesampling_1/Shape:output:0'sampling_1/strided_slice/stack:output:0)sampling_1/strided_slice/stack_1:output:0)sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
sampling_1/Shape_1Shapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:j
 sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:l
"sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
sampling_1/strided_slice_1StridedSlicesampling_1/Shape_1:output:0)sampling_1/strided_slice_1/stack:output:0+sampling_1/strided_slice_1/stack_1:output:0+sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
sampling_1/random_normal/shapePack!sampling_1/strided_slice:output:0#sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:b
sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    d
sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
-sampling_1/random_normal/RandomStandardNormalRandomStandardNormal'sampling_1/random_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2ۣ??
sampling_1/random_normal/mulMul6sampling_1/random_normal/RandomStandardNormal:output:0(sampling_1/random_normal/stddev:output:0*
T0*'
_output_shapes
:??????????
sampling_1/random_normalAddV2 sampling_1/random_normal/mul:z:0&sampling_1/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????U
sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?|
sampling_1/mulMulsampling_1/mul/x:output:0dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????[
sampling_1/ExpExpsampling_1/mul:z:0*
T0*'
_output_shapes
:?????????{
sampling_1/mul_1Mulsampling_1/Exp:y:0sampling_1/random_normal:z:0*
T0*'
_output_shapes
:?????????y
sampling_1/addAddV2dense_3/BiasAdd:output:0sampling_1/mul_1:z:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????i

Identity_1Identitydense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????c

Identity_2Identitysampling_1/add:z:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????d7: : : : : : : : 2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????d7
 
_user_specified_nameinputs
?
?
C__inference_decoder_layer_call_and_return_conditional_losses_207742

inputs9
&dense_5_matmul_readvariableop_resource:	?6
'dense_5_biasadd_readvariableop_resource:	?^
Hconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource:$@
2conv1d_transpose_2_biasadd_readvariableop_resource:$^
Hconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource:7$@
2conv1d_transpose_3_biasadd_readvariableop_resource:7
identity??)conv1d_transpose_2/BiasAdd/ReadVariableOp??conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?)conv1d_transpose_3/BiasAdd/ReadVariableOp??conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0z
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????Y
reshape_1/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
reshape_1/ReshapeReshapedense_5/Relu:activations:0 reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????b
conv1d_transpose_2/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:p
&conv1d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv1d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv1d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv1d_transpose_2/strided_sliceStridedSlice!conv1d_transpose_2/Shape:output:0/conv1d_transpose_2/strided_slice/stack:output:01conv1d_transpose_2/strided_slice/stack_1:output:01conv1d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
(conv1d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*conv1d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv1d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv1d_transpose_2/strided_slice_1StridedSlice!conv1d_transpose_2/Shape:output:01conv1d_transpose_2/strided_slice_1/stack:output:03conv1d_transpose_2/strided_slice_1/stack_1:output:03conv1d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv1d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv1d_transpose_2/mulMul+conv1d_transpose_2/strided_slice_1:output:0!conv1d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: Z
conv1d_transpose_2/add/yConst*
_output_shapes
: *
dtype0*
value	B : 
conv1d_transpose_2/addAddV2conv1d_transpose_2/mul:z:0!conv1d_transpose_2/add/y:output:0*
T0*
_output_shapes
: \
conv1d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :$?
conv1d_transpose_2/stackPack)conv1d_transpose_2/strided_slice:output:0conv1d_transpose_2/add:z:0#conv1d_transpose_2/stack/2:output:0*
N*
T0*
_output_shapes
:t
2conv1d_transpose_2/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
.conv1d_transpose_2/conv1d_transpose/ExpandDims
ExpandDimsreshape_1/Reshape:output:0;conv1d_transpose_2/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
dtype0v
4conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
0conv1d_transpose_2/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:$?
7conv1d_transpose_2/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1conv1d_transpose_2/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_2/stack:output:0@conv1d_transpose_2/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
9conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
;conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3conv1d_transpose_2/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_2/stack:output:0Bconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask}
3conv1d_transpose_2/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:q
/conv1d_transpose_2/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*conv1d_transpose_2/conv1d_transpose/concatConcatV2:conv1d_transpose_2/conv1d_transpose/strided_slice:output:0<conv1d_transpose_2/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_2/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_2/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
#conv1d_transpose_2/conv1d_transposeConv2DBackpropInput3conv1d_transpose_2/conv1d_transpose/concat:output:09conv1d_transpose_2/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_2/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????2$*
paddingVALID*
strides
?
+conv1d_transpose_2/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_2/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims
?
)conv1d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
conv1d_transpose_2/BiasAddBiasAdd4conv1d_transpose_2/conv1d_transpose/Squeeze:output:01conv1d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$z
conv1d_transpose_2/ReluRelu#conv1d_transpose_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$m
conv1d_transpose_3/ShapeShape%conv1d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:p
&conv1d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv1d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv1d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv1d_transpose_3/strided_sliceStridedSlice!conv1d_transpose_3/Shape:output:0/conv1d_transpose_3/strided_slice/stack:output:01conv1d_transpose_3/strided_slice/stack_1:output:01conv1d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
(conv1d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*conv1d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv1d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv1d_transpose_3/strided_slice_1StridedSlice!conv1d_transpose_3/Shape:output:01conv1d_transpose_3/strided_slice_1/stack:output:03conv1d_transpose_3/strided_slice_1/stack_1:output:03conv1d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv1d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
conv1d_transpose_3/mulMul+conv1d_transpose_3/strided_slice_1:output:0!conv1d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: Z
conv1d_transpose_3/add/yConst*
_output_shapes
: *
dtype0*
value	B : 
conv1d_transpose_3/addAddV2conv1d_transpose_3/mul:z:0!conv1d_transpose_3/add/y:output:0*
T0*
_output_shapes
: \
conv1d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :7?
conv1d_transpose_3/stackPack)conv1d_transpose_3/strided_slice:output:0conv1d_transpose_3/add:z:0#conv1d_transpose_3/stack/2:output:0*
N*
T0*
_output_shapes
:t
2conv1d_transpose_3/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
.conv1d_transpose_3/conv1d_transpose/ExpandDims
ExpandDims%conv1d_transpose_2/Relu:activations:0;conv1d_transpose_3/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2$?
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpHconv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
dtype0v
4conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
0conv1d_transpose_3/conv1d_transpose/ExpandDims_1
ExpandDimsGconv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0=conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:7$?
7conv1d_transpose_3/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1conv1d_transpose_3/conv1d_transpose/strided_sliceStridedSlice!conv1d_transpose_3/stack:output:0@conv1d_transpose_3/conv1d_transpose/strided_slice/stack:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice/stack_1:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
9conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
;conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3conv1d_transpose_3/conv1d_transpose/strided_slice_1StridedSlice!conv1d_transpose_3/stack:output:0Bconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack:output:0Dconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1:output:0Dconv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask}
3conv1d_transpose_3/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:q
/conv1d_transpose_3/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
*conv1d_transpose_3/conv1d_transpose/concatConcatV2:conv1d_transpose_3/conv1d_transpose/strided_slice:output:0<conv1d_transpose_3/conv1d_transpose/concat/values_1:output:0<conv1d_transpose_3/conv1d_transpose/strided_slice_1:output:08conv1d_transpose_3/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
#conv1d_transpose_3/conv1d_transposeConv2DBackpropInput3conv1d_transpose_3/conv1d_transpose/concat:output:09conv1d_transpose_3/conv1d_transpose/ExpandDims_1:output:07conv1d_transpose_3/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d7*
paddingVALID*
strides
?
+conv1d_transpose_3/conv1d_transpose/SqueezeSqueeze,conv1d_transpose_3/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????d7*
squeeze_dims
?
)conv1d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv1d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0?
conv1d_transpose_3/BiasAddBiasAdd4conv1d_transpose_3/conv1d_transpose/Squeeze:output:01conv1d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d7z
conv1d_transpose_3/ReluRelu#conv1d_transpose_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d7x
IdentityIdentity%conv1d_transpose_3/Relu:activations:0^NoOp*
T0*+
_output_shapes
:?????????d7?
NoOpNoOp*^conv1d_transpose_2/BiasAdd/ReadVariableOp@^conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp*^conv1d_transpose_3/BiasAdd/ReadVariableOp@^conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2V
)conv1d_transpose_2/BiasAdd/ReadVariableOp)conv1d_transpose_2/BiasAdd/ReadVariableOp2?
?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2V
)conv1d_transpose_3/BiasAdd/ReadVariableOp)conv1d_transpose_3/BiasAdd/ReadVariableOp2?
?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
G__inference_autoencoder_layer_call_and_return_conditional_losses_207539

inputsR
<encoder_conv1d_2_conv1d_expanddims_1_readvariableop_resource:7$>
0encoder_conv1d_2_biasadd_readvariableop_resource:$R
<encoder_conv1d_3_conv1d_expanddims_1_readvariableop_resource:$>
0encoder_conv1d_3_biasadd_readvariableop_resource:A
.encoder_dense_3_matmul_readvariableop_resource:	?=
/encoder_dense_3_biasadd_readvariableop_resource:A
.encoder_dense_4_matmul_readvariableop_resource:	?=
/encoder_dense_4_biasadd_readvariableop_resource:A
.decoder_dense_5_matmul_readvariableop_resource:	?>
/decoder_dense_5_biasadd_readvariableop_resource:	?f
Pdecoder_conv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource:$H
:decoder_conv1d_transpose_2_biasadd_readvariableop_resource:$f
Pdecoder_conv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource:7$H
:decoder_conv1d_transpose_3_biasadd_readvariableop_resource:7
identity

identity_1??1decoder/conv1d_transpose_2/BiasAdd/ReadVariableOp?Gdecoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp?1decoder/conv1d_transpose_3/BiasAdd/ReadVariableOp?Gdecoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp?&decoder/dense_5/BiasAdd/ReadVariableOp?%decoder/dense_5/MatMul/ReadVariableOp?'encoder/conv1d_2/BiasAdd/ReadVariableOp?3encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp?'encoder/conv1d_3/BiasAdd/ReadVariableOp?3encoder/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp?&encoder/dense_3/BiasAdd/ReadVariableOp?%encoder/dense_3/MatMul/ReadVariableOp?&encoder/dense_4/BiasAdd/ReadVariableOp?%encoder/dense_4/MatMul/ReadVariableOpq
&encoder/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"encoder/conv1d_2/Conv1D/ExpandDims
ExpandDimsinputs/encoder/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d7?
3encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp<encoder_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
dtype0j
(encoder/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
$encoder/conv1d_2/Conv1D/ExpandDims_1
ExpandDims;encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:01encoder/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:7$?
encoder/conv1d_2/Conv1DConv2D+encoder/conv1d_2/Conv1D/ExpandDims:output:0-encoder/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????2$*
paddingVALID*
strides
?
encoder/conv1d_2/Conv1D/SqueezeSqueeze encoder/conv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims

??????????
'encoder/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
encoder/conv1d_2/BiasAddBiasAdd(encoder/conv1d_2/Conv1D/Squeeze:output:0/encoder/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$v
encoder/conv1d_2/ReluRelu!encoder/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$q
&encoder/conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"encoder/conv1d_3/Conv1D/ExpandDims
ExpandDims#encoder/conv1d_2/Relu:activations:0/encoder/conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2$?
3encoder/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp<encoder_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
dtype0j
(encoder/conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
$encoder/conv1d_3/Conv1D/ExpandDims_1
ExpandDims;encoder/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:01encoder/conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:$?
encoder/conv1d_3/Conv1DConv2D+encoder/conv1d_3/Conv1D/ExpandDims:output:0-encoder/conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
encoder/conv1d_3/Conv1D/SqueezeSqueeze encoder/conv1d_3/Conv1D:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

??????????
'encoder/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
encoder/conv1d_3/BiasAddBiasAdd(encoder/conv1d_3/Conv1D/Squeeze:output:0/encoder/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????v
encoder/conv1d_3/ReluRelu!encoder/conv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????h
encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
encoder/flatten_1/ReshapeReshape#encoder/conv1d_3/Relu:activations:0 encoder/flatten_1/Const:output:0*
T0*(
_output_shapes
:???????????
%encoder/dense_3/MatMul/ReadVariableOpReadVariableOp.encoder_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
encoder/dense_3/MatMulMatMul"encoder/flatten_1/Reshape:output:0-encoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&encoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
encoder/dense_3/BiasAddBiasAdd encoder/dense_3/MatMul:product:0.encoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
%encoder/dense_4/MatMul/ReadVariableOpReadVariableOp.encoder_dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
encoder/dense_4/MatMulMatMul"encoder/flatten_1/Reshape:output:0-encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
encoder/dense_4/BiasAddBiasAdd encoder/dense_4/MatMul:product:0.encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
encoder/sampling_1/ShapeShape encoder/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:p
&encoder/sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(encoder/sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(encoder/sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 encoder/sampling_1/strided_sliceStridedSlice!encoder/sampling_1/Shape:output:0/encoder/sampling_1/strided_slice/stack:output:01encoder/sampling_1/strided_slice/stack_1:output:01encoder/sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
encoder/sampling_1/Shape_1Shape encoder/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:r
(encoder/sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*encoder/sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*encoder/sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"encoder/sampling_1/strided_slice_1StridedSlice#encoder/sampling_1/Shape_1:output:01encoder/sampling_1/strided_slice_1/stack:output:03encoder/sampling_1/strided_slice_1/stack_1:output:03encoder/sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
&encoder/sampling_1/random_normal/shapePack)encoder/sampling_1/strided_slice:output:0+encoder/sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:j
%encoder/sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    l
'encoder/sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
5encoder/sampling_1/random_normal/RandomStandardNormalRandomStandardNormal/encoder/sampling_1/random_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2????
$encoder/sampling_1/random_normal/mulMul>encoder/sampling_1/random_normal/RandomStandardNormal:output:00encoder/sampling_1/random_normal/stddev:output:0*
T0*'
_output_shapes
:??????????
 encoder/sampling_1/random_normalAddV2(encoder/sampling_1/random_normal/mul:z:0.encoder/sampling_1/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????]
encoder/sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
encoder/sampling_1/mulMul!encoder/sampling_1/mul/x:output:0 encoder/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????k
encoder/sampling_1/ExpExpencoder/sampling_1/mul:z:0*
T0*'
_output_shapes
:??????????
encoder/sampling_1/mul_1Mulencoder/sampling_1/Exp:y:0$encoder/sampling_1/random_normal:z:0*
T0*'
_output_shapes
:??????????
encoder/sampling_1/addAddV2 encoder/dense_3/BiasAdd:output:0encoder/sampling_1/mul_1:z:0*
T0*'
_output_shapes
:??????????
%decoder/dense_5/MatMul/ReadVariableOpReadVariableOp.decoder_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
decoder/dense_5/MatMulMatMulencoder/sampling_1/add:z:0-decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
decoder/dense_5/BiasAddBiasAdd decoder/dense_5/MatMul:product:0.decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????q
decoder/dense_5/ReluRelu decoder/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????i
decoder/reshape_1/ShapeShape"decoder/dense_5/Relu:activations:0*
T0*
_output_shapes
:o
%decoder/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'decoder/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'decoder/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
decoder/reshape_1/strided_sliceStridedSlice decoder/reshape_1/Shape:output:0.decoder/reshape_1/strided_slice/stack:output:00decoder/reshape_1/strided_slice/stack_1:output:00decoder/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!decoder/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!decoder/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
decoder/reshape_1/Reshape/shapePack(decoder/reshape_1/strided_slice:output:0*decoder/reshape_1/Reshape/shape/1:output:0*decoder/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
decoder/reshape_1/ReshapeReshape"decoder/dense_5/Relu:activations:0(decoder/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????r
 decoder/conv1d_transpose_2/ShapeShape"decoder/reshape_1/Reshape:output:0*
T0*
_output_shapes
:x
.decoder/conv1d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv1d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv1d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv1d_transpose_2/strided_sliceStridedSlice)decoder/conv1d_transpose_2/Shape:output:07decoder/conv1d_transpose_2/strided_slice/stack:output:09decoder/conv1d_transpose_2/strided_slice/stack_1:output:09decoder/conv1d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
0decoder/conv1d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv1d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv1d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/conv1d_transpose_2/strided_slice_1StridedSlice)decoder/conv1d_transpose_2/Shape:output:09decoder/conv1d_transpose_2/strided_slice_1/stack:output:0;decoder/conv1d_transpose_2/strided_slice_1/stack_1:output:0;decoder/conv1d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 decoder/conv1d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
decoder/conv1d_transpose_2/mulMul3decoder/conv1d_transpose_2/strided_slice_1:output:0)decoder/conv1d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: b
 decoder/conv1d_transpose_2/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
decoder/conv1d_transpose_2/addAddV2"decoder/conv1d_transpose_2/mul:z:0)decoder/conv1d_transpose_2/add/y:output:0*
T0*
_output_shapes
: d
"decoder/conv1d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :$?
 decoder/conv1d_transpose_2/stackPack1decoder/conv1d_transpose_2/strided_slice:output:0"decoder/conv1d_transpose_2/add:z:0+decoder/conv1d_transpose_2/stack/2:output:0*
N*
T0*
_output_shapes
:|
:decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
6decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims
ExpandDims"decoder/reshape_1/Reshape:output:0Cdecoder/conv1d_transpose_2/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:??????????
Gdecoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpPdecoder_conv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:$*
dtype0~
<decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
8decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1
ExpandDimsOdecoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Edecoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:$?
?decoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Adecoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Adecoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9decoder/conv1d_transpose_2/conv1d_transpose/strided_sliceStridedSlice)decoder/conv1d_transpose_2/stack:output:0Hdecoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack:output:0Jdecoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1:output:0Jdecoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Adecoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cdecoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Cdecoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1StridedSlice)decoder/conv1d_transpose_2/stack:output:0Jdecoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack:output:0Ldecoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1:output:0Ldecoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
;decoder/conv1d_transpose_2/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:y
7decoder/conv1d_transpose_2/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2decoder/conv1d_transpose_2/conv1d_transpose/concatConcatV2Bdecoder/conv1d_transpose_2/conv1d_transpose/strided_slice:output:0Ddecoder/conv1d_transpose_2/conv1d_transpose/concat/values_1:output:0Ddecoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1:output:0@decoder/conv1d_transpose_2/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
+decoder/conv1d_transpose_2/conv1d_transposeConv2DBackpropInput;decoder/conv1d_transpose_2/conv1d_transpose/concat:output:0Adecoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1:output:0?decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????2$*
paddingVALID*
strides
?
3decoder/conv1d_transpose_2/conv1d_transpose/SqueezeSqueeze4decoder/conv1d_transpose_2/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????2$*
squeeze_dims
?
1decoder/conv1d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv1d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
"decoder/conv1d_transpose_2/BiasAddBiasAdd<decoder/conv1d_transpose_2/conv1d_transpose/Squeeze:output:09decoder/conv1d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2$?
decoder/conv1d_transpose_2/ReluRelu+decoder/conv1d_transpose_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2$}
 decoder/conv1d_transpose_3/ShapeShape-decoder/conv1d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv1d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv1d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv1d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv1d_transpose_3/strided_sliceStridedSlice)decoder/conv1d_transpose_3/Shape:output:07decoder/conv1d_transpose_3/strided_slice/stack:output:09decoder/conv1d_transpose_3/strided_slice/stack_1:output:09decoder/conv1d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
0decoder/conv1d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv1d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv1d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/conv1d_transpose_3/strided_slice_1StridedSlice)decoder/conv1d_transpose_3/Shape:output:09decoder/conv1d_transpose_3/strided_slice_1/stack:output:0;decoder/conv1d_transpose_3/strided_slice_1/stack_1:output:0;decoder/conv1d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 decoder/conv1d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :?
decoder/conv1d_transpose_3/mulMul3decoder/conv1d_transpose_3/strided_slice_1:output:0)decoder/conv1d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: b
 decoder/conv1d_transpose_3/add/yConst*
_output_shapes
: *
dtype0*
value	B : ?
decoder/conv1d_transpose_3/addAddV2"decoder/conv1d_transpose_3/mul:z:0)decoder/conv1d_transpose_3/add/y:output:0*
T0*
_output_shapes
: d
"decoder/conv1d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :7?
 decoder/conv1d_transpose_3/stackPack1decoder/conv1d_transpose_3/strided_slice:output:0"decoder/conv1d_transpose_3/add:z:0+decoder/conv1d_transpose_3/stack/2:output:0*
N*
T0*
_output_shapes
:|
:decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
6decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims
ExpandDims-decoder/conv1d_transpose_2/Relu:activations:0Cdecoder/conv1d_transpose_3/conv1d_transpose/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2$?
Gdecoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpPdecoder_conv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:7$*
dtype0~
<decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
8decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1
ExpandDimsOdecoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Edecoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:7$?
?decoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Adecoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Adecoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9decoder/conv1d_transpose_3/conv1d_transpose/strided_sliceStridedSlice)decoder/conv1d_transpose_3/stack:output:0Hdecoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack:output:0Jdecoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1:output:0Jdecoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Adecoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Cdecoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Cdecoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1StridedSlice)decoder/conv1d_transpose_3/stack:output:0Jdecoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack:output:0Ldecoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1:output:0Ldecoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask?
;decoder/conv1d_transpose_3/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:y
7decoder/conv1d_transpose_3/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2decoder/conv1d_transpose_3/conv1d_transpose/concatConcatV2Bdecoder/conv1d_transpose_3/conv1d_transpose/strided_slice:output:0Ddecoder/conv1d_transpose_3/conv1d_transpose/concat/values_1:output:0Ddecoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1:output:0@decoder/conv1d_transpose_3/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:?
+decoder/conv1d_transpose_3/conv1d_transposeConv2DBackpropInput;decoder/conv1d_transpose_3/conv1d_transpose/concat:output:0Adecoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1:output:0?decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims:output:0*
T0*/
_output_shapes
:?????????d7*
paddingVALID*
strides
?
3decoder/conv1d_transpose_3/conv1d_transpose/SqueezeSqueeze4decoder/conv1d_transpose_3/conv1d_transpose:output:0*
T0*+
_output_shapes
:?????????d7*
squeeze_dims
?
1decoder/conv1d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv1d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:7*
dtype0?
"decoder/conv1d_transpose_3/BiasAddBiasAdd<decoder/conv1d_transpose_3/conv1d_transpose/Squeeze:output:09decoder/conv1d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d7?
decoder/conv1d_transpose_3/ReluRelu+decoder/conv1d_transpose_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d7d
SquareSquare encoder/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????j
subSub encoder/dense_4/BiasAdd:output:0
Square:y:0*
T0*'
_output_shapes
:?????????^
ExpExp encoder/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????P
sub_1Subsub:z:0Exp:y:0*
T0*'
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??Y
addAddV2	sub_1:z:0add/y:output:0*
T0*'
_output_shapes
:?????????V
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
IdentityIdentity-decoder/conv1d_transpose_3/Relu:activations:0^NoOp*
T0*+
_output_shapes
:?????????d7G

Identity_1Identitymul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp2^decoder/conv1d_transpose_2/BiasAdd/ReadVariableOpH^decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2^decoder/conv1d_transpose_3/BiasAdd/ReadVariableOpH^decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp'^decoder/dense_5/BiasAdd/ReadVariableOp&^decoder/dense_5/MatMul/ReadVariableOp(^encoder/conv1d_2/BiasAdd/ReadVariableOp4^encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp(^encoder/conv1d_3/BiasAdd/ReadVariableOp4^encoder/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp'^encoder/dense_3/BiasAdd/ReadVariableOp&^encoder/dense_3/MatMul/ReadVariableOp'^encoder/dense_4/BiasAdd/ReadVariableOp&^encoder/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????d7: : : : : : : : : : : : : : 2f
1decoder/conv1d_transpose_2/BiasAdd/ReadVariableOp1decoder/conv1d_transpose_2/BiasAdd/ReadVariableOp2?
Gdecoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpGdecoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2f
1decoder/conv1d_transpose_3/BiasAdd/ReadVariableOp1decoder/conv1d_transpose_3/BiasAdd/ReadVariableOp2?
Gdecoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpGdecoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp2P
&decoder/dense_5/BiasAdd/ReadVariableOp&decoder/dense_5/BiasAdd/ReadVariableOp2N
%decoder/dense_5/MatMul/ReadVariableOp%decoder/dense_5/MatMul/ReadVariableOp2R
'encoder/conv1d_2/BiasAdd/ReadVariableOp'encoder/conv1d_2/BiasAdd/ReadVariableOp2j
3encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp3encoder/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2R
'encoder/conv1d_3/BiasAdd/ReadVariableOp'encoder/conv1d_3/BiasAdd/ReadVariableOp2j
3encoder/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp3encoder/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2P
&encoder/dense_3/BiasAdd/ReadVariableOp&encoder/dense_3/BiasAdd/ReadVariableOp2N
%encoder/dense_3/MatMul/ReadVariableOp%encoder/dense_3/MatMul/ReadVariableOp2P
&encoder/dense_4/BiasAdd/ReadVariableOp&encoder/dense_4/BiasAdd/ReadVariableOp2N
%encoder/dense_4/MatMul/ReadVariableOp%encoder/dense_4/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????d7
 
_user_specified_nameinputs
?=
?
"__inference__traced_restore_207961
file_prefixJ
4assignvariableop_autoencoder_encoder_conv1d_2_kernel:7$B
4assignvariableop_1_autoencoder_encoder_conv1d_2_bias:$L
6assignvariableop_2_autoencoder_encoder_conv1d_3_kernel:$B
4assignvariableop_3_autoencoder_encoder_conv1d_3_bias:H
5assignvariableop_4_autoencoder_encoder_dense_3_kernel:	?A
3assignvariableop_5_autoencoder_encoder_dense_3_bias:H
5assignvariableop_6_autoencoder_encoder_dense_4_kernel:	?A
3assignvariableop_7_autoencoder_encoder_dense_4_bias:H
5assignvariableop_8_autoencoder_decoder_dense_5_kernel:	?B
3assignvariableop_9_autoencoder_decoder_dense_5_bias:	?W
Aassignvariableop_10_autoencoder_decoder_conv1d_transpose_2_kernel:$M
?assignvariableop_11_autoencoder_decoder_conv1d_transpose_2_bias:$W
Aassignvariableop_12_autoencoder_decoder_conv1d_transpose_3_kernel:7$M
?assignvariableop_13_autoencoder_decoder_conv1d_transpose_3_bias:7
identity_15??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp4assignvariableop_autoencoder_encoder_conv1d_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp4assignvariableop_1_autoencoder_encoder_conv1d_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp6assignvariableop_2_autoencoder_encoder_conv1d_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp4assignvariableop_3_autoencoder_encoder_conv1d_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_autoencoder_encoder_dense_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp3assignvariableop_5_autoencoder_encoder_dense_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp5assignvariableop_6_autoencoder_encoder_dense_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp3assignvariableop_7_autoencoder_encoder_dense_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp5assignvariableop_8_autoencoder_decoder_dense_5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp3assignvariableop_9_autoencoder_decoder_dense_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpAassignvariableop_10_autoencoder_decoder_conv1d_transpose_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp?assignvariableop_11_autoencoder_decoder_conv1d_transpose_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpAassignvariableop_12_autoencoder_decoder_conv1d_transpose_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp?assignvariableop_13_autoencoder_decoder_conv1d_transpose_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
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
?
?
(__inference_decoder_layer_call_fn_207646

inputs
unknown:	?
	unknown_0:	?
	unknown_1:$
	unknown_2:$
	unknown_3:7$
	unknown_4:7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d7*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_207121s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????d7`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
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
serving_default_input_1:0?????????d7@
output_14
StatefulPartitionedCall:0?????????d7tensorflow/serving/predict:Ś
?
encoder
decoder
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_model
?
	conv0
		conv1

flat

dense_mean
dense_log_var
sampling
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	dense
reshape
	conv1
	conv2
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
&12
'13"
trackable_list_wrapper
?
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
&12
'13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

kernel
bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

 kernel
!bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
X
0
1
2
3
4
5
 6
!7"
trackable_list_wrapper
X
0
1
2
3
4
5
 6
!7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

"kernel
#bias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

$kernel
%bias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

&kernel
'bias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
J
"0
#1
$2
%3
&4
'5"
trackable_list_wrapper
J
"0
#1
$2
%3
&4
'5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
9:77$2#autoencoder/encoder/conv1d_2/kernel
/:-$2!autoencoder/encoder/conv1d_2/bias
9:7$2#autoencoder/encoder/conv1d_3/kernel
/:-2!autoencoder/encoder/conv1d_3/bias
5:3	?2"autoencoder/encoder/dense_3/kernel
.:,2 autoencoder/encoder/dense_3/bias
5:3	?2"autoencoder/encoder/dense_4/kernel
.:,2 autoencoder/encoder/dense_4/bias
5:3	?2"autoencoder/decoder/dense_5/kernel
/:-?2 autoencoder/decoder/dense_5/bias
C:A$2-autoencoder/decoder/conv1d_transpose_2/kernel
9:7$2+autoencoder/decoder/conv1d_transpose_2/bias
C:A7$2-autoencoder/decoder/conv1d_transpose_3/kernel
9:772+autoencoder/decoder/conv1d_transpose_3/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
-	variables
.trainable_variables
/regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
1	variables
2trainable_variables
3regularization_losses
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
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
5	variables
6trainable_variables
7regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
9	variables
:trainable_variables
;regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
=	variables
>trainable_variables
?regularization_losses
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
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
0
	1

2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
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
N	variables
Otrainable_variables
Pregularization_losses
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
2
3"
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
?2?
,__inference_autoencoder_layer_call_fn_207179
,__inference_autoencoder_layer_call_fn_207373?
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
G__inference_autoencoder_layer_call_and_return_conditional_losses_207539
G__inference_autoencoder_layer_call_and_return_conditional_losses_207304?
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
!__inference__wrapped_model_206827input_1"?
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
(__inference_encoder_layer_call_fn_207564?
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
C__inference_encoder_layer_call_and_return_conditional_losses_207629?
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
(__inference_decoder_layer_call_fn_207646?
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
C__inference_decoder_layer_call_and_return_conditional_losses_207742?
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
$__inference_signature_wrapper_207339input_1"?
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
3__inference_conv1d_transpose_2_layer_call_fn_207751?
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
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_207793?
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
3__inference_conv1d_transpose_3_layer_call_fn_207802?
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
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_207844?
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
!__inference__wrapped_model_206827 !"#$%&'4?1
*?'
%?"
input_1?????????d7
? "7?4
2
output_1&?#
output_1?????????d7?
G__inference_autoencoder_layer_call_and_return_conditional_losses_207304 !"#$%&'4?1
*?'
%?"
input_1?????????d7
? "7?4
?
0?????????d7
?
?	
1/0 ?
G__inference_autoencoder_layer_call_and_return_conditional_losses_207539~ !"#$%&'3?0
)?&
$?!
inputs?????????d7
? "7?4
?
0?????????d7
?
?	
1/0 ?
,__inference_autoencoder_layer_call_fn_207179d !"#$%&'4?1
*?'
%?"
input_1?????????d7
? "??????????d7?
,__inference_autoencoder_layer_call_fn_207373c !"#$%&'3?0
)?&
$?!
inputs?????????d7
? "??????????d7?
N__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_207793v$%<?9
2?/
-?*
inputs??????????????????
? "2?/
(?%
0??????????????????$
? ?
3__inference_conv1d_transpose_2_layer_call_fn_207751i$%<?9
2?/
-?*
inputs??????????????????
? "%?"??????????????????$?
N__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_207844v&'<?9
2?/
-?*
inputs??????????????????$
? "2?/
(?%
0??????????????????7
? ?
3__inference_conv1d_transpose_3_layer_call_fn_207802i&'<?9
2?/
-?*
inputs??????????????????$
? "%?"??????????????????7?
C__inference_decoder_layer_call_and_return_conditional_losses_207742d"#$%&'/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????d7
? ?
(__inference_decoder_layer_call_fn_207646W"#$%&'/?,
%?"
 ?
inputs?????????
? "??????????d7?
C__inference_encoder_layer_call_and_return_conditional_losses_207629? !3?0
)?&
$?!
inputs?????????d7
? "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
(__inference_encoder_layer_call_fn_207564? !3?0
)?&
$?!
inputs?????????d7
? "Z?W
?
0?????????
?
1?????????
?
2??????????
$__inference_signature_wrapper_207339? !"#$%&'??<
? 
5?2
0
input_1%?"
input_1?????????d7"7?4
2
output_1&?#
output_1?????????d7