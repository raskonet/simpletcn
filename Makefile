# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -pthread

# Project files - ADD utils.cpp HERE
SRCS = main.cpp tcn.cpp residual_block.cpp conv1d.cpp dropout.cpp relu.cpp tensor.cpp utils.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = tcn_app

# Build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Test targets
test_conv: conv1d_test.cpp conv1d.cpp tensor.cpp
	$(CXX) $(CXXFLAGS) -o test_conv conv1d_test.cpp conv1d.cpp tensor.cpp
	./test_conv

test_dropout: dropout_test.cpp dropout.cpp tensor.cpp
	$(CXX) $(CXXFLAGS) -o test_dropout dropout_test.cpp dropout.cpp tensor.cpp
	./test_dropout

test_relu: relu_test.cpp relu.cpp tensor.cpp
	$(CXX) $(CXXFLAGS) -o test_relu relu_test.cpp relu.cpp tensor.cpp
	./test_relu

test_res: residual_block_test.cpp residual_block.cpp conv1d.cpp dropout.cpp relu.cpp tensor.cpp
	$(CXX) $(CXXFLAGS) -o test_res residual_block_test.cpp residual_block.cpp conv1d.cpp dropout.cpp relu.cpp tensor.cpp
	./test_res

test_tcn: tcn_test.cpp tcn.cpp residual_block.cpp conv1d.cpp dropout.cpp relu.cpp tensor.cpp
	$(CXX) $(CXXFLAGS) -o test_tcn tcn_test.cpp tcn.cpp residual_block.cpp conv1d.cpp dropout.cpp relu.cpp tensor.cpp
	./test_tcn

clean:
	rm -f $(OBJS) $(TARGET) test_conv test_dropout test_relu test_res test_tcn

.PHONY: all clean test_conv test_dropout test_relu test_res test_tcn
