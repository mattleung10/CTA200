CXX = g++
CXXFLAGS = -Wall -O2
LDFLAGS = -lm

all: main

main: main.o
	$(CXX) -o $@ $< $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	$(RM) main *.o
