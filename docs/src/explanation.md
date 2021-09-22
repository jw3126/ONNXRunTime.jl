# Explanation

## Memory layout

onnxruntime expects tensors in with memory in C-layout. The julia `Array` type is in Fortran-layout however.
The high level api automagically handles the layout conversions between julia and onnxruntime.
In the low level api conversions are only done if explicitly documented.

For instance consider the "copy2d.onnx", which just returns a copy of its input:
```julia
using ONNXRunTime
path = ONNXRunTime.testdatapath("copy2d.onnx")
model = load_inference(path);
x = [1 2 3; 4 5 6]
y = model((input=x,)).output
@test x == y
```
Under the hood the following happens. We start with the julia matrix:
```julia
# julia : [1 2 3;4 5 6]
# memory: 1 4 2 5 3 6
```
As a preproccing step this array is copied to C layout:
```julia
# julia : [1 3 5; 2 4 6]
# memory: 1 2 3 4 5 6
```
Then onnxruntime is invoked producing an output in C layout:
```julia
# julia : [1 3 5; 2 4 6]
# memory: 1 2 3 4 5 6
```
Finally as a post processing step this converted to fortran layout:
```julia
# julia [1 2 3; 4 5 6]
# memory: 1 4 2 5 3 6
```
