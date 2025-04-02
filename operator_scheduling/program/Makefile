CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -Ijson/include
TARGET = main.out
SRC = main.cpp solve.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
