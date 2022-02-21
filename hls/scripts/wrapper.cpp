#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include <inttypes.h>
#include <math.h>

#define N 11
#define N2 121 // N*N
#define M 6
#define M2 36
#define DWIDTH 512

extern "C" void forward(float (&arg_2)[1][1][11][11], float (&arg_3)[1][2]);

typedef ap_axiu<DWIDTH, 0, 0, 0> axis_t;

typedef float DataType;

// bit width of data type (32)
const int DataTypeSize = sizeof(DataType) * 8;
// number of floats that fit into the stream (16)
const int NUM_ITEMS = DWIDTH / DataTypeSize;

typedef ap_uint<DataTypeSize> DataTypeInt;
typedef union converter {
  DataType d;
  uint32_t i;
} converter_t;

extern "C" {
void wrapper(hls::stream<axis_t> &in, hls::stream<axis_t> &out) {
#pragma HLS INTERFACE axis port = in
#pragma HLS INTERFACE axis port = out

  float l_A[1][1][11][11];

  //#pragma HLS ARRAY_PARTITION variable = l_A factor = 16 dim = 1 cyclic
  //#pragma HLS ARRAY_PARTITION variable = l_C factor = 16 dim = 1 cyclic

  // how many stream reads it takes to get the entire matrix
  int i_limit = ceil((float)(11 * 11) / (float)NUM_ITEMS);
  converter_t converter;

load_A:
  for (int i = 0; i < i_limit; i++) {
    axis_t temp = in.read();
    for (int j = 0; j < NUM_ITEMS; j++) {
      int high = (j + 1) * DataTypeSize - 1;
      int low = j * DataTypeSize;
      int index = i * NUM_ITEMS + j;

      converter.i = temp.data.range(high, low);
      (&l_A[0][0][0][0])[index] = converter.d;
    }
  }

  float l_C[1][2];
  forward(l_A, l_C);

  // 36 / 16 = 2 but it should be 3
  //  int k_limit = ceil((float)M2 / (float)NUM_ITEMS);
  int k_limit = ceil((float)(1 * 2) / (float)NUM_ITEMS);
writeC:
  for (int k = 0; k < k_limit; k++) {
    axis_t temp;
    for (int j = 0; j < NUM_ITEMS; j++) {
      int high = (j + 1) * DataTypeSize - 1;
      int low = j * DataTypeSize;
      int index = k * NUM_ITEMS + j;
      converter.d = (&l_C[0][0])[index];
      temp.data.range(high, low) = converter.i;
    }

    ap_uint<1> last = 0;
    if (k == k_limit - 1) {
      last = 1;
    }
    temp.last = last;
    temp.keep = -1; // enabling all bytes
    out.write(temp);
  }
}
}
