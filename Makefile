CC=gcc
CFLAGS=-Wall -std=c++11 -O3 -g -fPIC

#TODO: auto-download, or search common paths
HDF5_LIB=-L/home/skovaka/anaconda3/lib /home/skovaka/anaconda3/lib/libhdf5.a
HDF5_INCLUDE=-I/home/skovaka/anaconda3/include

BWA_LIB=-L bwa bwa/libbwa.a

LIBS=$(HDF5_LIB) $(BWA_LIB) -lstdc++ -lz -ldl -pthread -lm 
INCLUDE=-I toml11 -I fast5/include -I pybind11/include -I pdqsort $(HDF5_INCLUDE)

SRC=src
BUILD=build
BIN=bin

_COMMON_OBJS=mapper.o seed_tracker.o range.o event_detector.o normalizer.o chunk.o read_buffer.o fast5_reader.o


_MAP_OBJS=$(_COMMON_OBJS) map_pool.o uncalled_map.o 
_SIM_OBJS=$(_COMMON_OBJS) realtime_pool.o client_sim.o uncalled_sim.o 
_DTW_OBJS=dtw_test.o fast5_reader.o read_buffer.o normalizer.o chunk.o event_detector.o range.o

_ALL_OBJS=$(_COMMON_OBJS) realtime_pool.o map_pool.o uncalled_map.o client_sim.o uncalled_sim.o dtw_test.o

MAP_OBJS = $(patsubst %, $(BUILD)/%, $(_MAP_OBJS))
SIM_OBJS = $(patsubst %, $(BUILD)/%, $(_SIM_OBJS))
DTW_OBJS = $(patsubst %, $(BUILD)/%, $(_DTW_OBJS))
ALL_OBJS = $(patsubst %, $(BUILD)/%, $(_ALL_OBJS))

DEPENDS := $(patsubst %.o, %.d, $(ALL_OBJS))

MAP_BIN = $(BIN)/uncalled_map
SIM_BIN = $(BIN)/uncalled_sim
DTW_BIN = $(BIN)/dtw_test

all: $(MAP_BIN) $(SIM_BIN) $(DTW_BIN)

$(MAP_BIN): $(MAP_OBJS) 
	$(CC) $(CFLAGS) $(MAP_OBJS) -o $@ $(LIBS)

$(SIM_BIN): $(SIM_OBJS) 
	$(CC) $(CFLAGS) $(SIM_OBJS) -o $@ $(LIBS)

$(DTW_BIN): $(DTW_OBJS)
	$(CC) $(CFLAGS) $(DTW_OBJS) -o $@ $(LIBS)

-include $(DEPENDS)

$(BUILD)/%.o: $(SRC)/%.cpp
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@ $(INCLUDE)

clean:
	rm $(BUILD)/*
