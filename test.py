import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import os

# Add the path to cl.exe in the PATH environment variable
#os.environ['PATH'] += r";C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64"

mod = SourceModule("""
__global__ void my_kernel() { }
""", options=["-arch=sm_89"])

print("PyCUDA is working with sm_89 architecture!")
