 swig -c++ -python osc.i
 g++ -c -fpic osc_wrap.cxx osc.cxx -I/usr/include/python3.7
 g++ -shared osc.o osc_wrap.o -o _oscillator_cpp.so