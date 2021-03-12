CXX = g++-7
FLAGS = -std=gnu++14  

TARGETS = main

all: $(TARGETS)

main: main.cpp
	$(CXX) $(FLAGS) $< -o $@ $(LIBS)

clean:
	rm $(TARGETS)