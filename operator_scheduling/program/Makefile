CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -Ijson/include
MAIN_TARGET = main.out
EVALUATOR_TARGET = evaluator.out
MAIN_SRC = main.cpp solver.cpp
EVALUATOR_SRC = evaluator.cpp solver.cpp

all: $(MAIN_TARGET) $(EVALUATOR_TARGET)

$(MAIN_TARGET): $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) -o $(MAIN_TARGET) $(MAIN_SRC)

$(EVALUATOR_TARGET): $(EVALUATOR_SRC)
	$(CXX) $(CXXFLAGS) -o $(EVALUATOR_TARGET) $(EVALUATOR_SRC)

clean:
	rm -f $(MAIN_TARGET) $(EVALUATOR_TARGET)
