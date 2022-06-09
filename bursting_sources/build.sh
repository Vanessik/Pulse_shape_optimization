 swig -c++ -python gfg.i
 g++ -c -fpic gfg_wrap.cxx gfg.cxx -I/usr/include/python3.7
 g++ -shared gfg.o gfg_wrap.o -o _oscillator_bursting.so