#include "ap_axi_sdata.h"
#include "ap_int.h"
#include <inttypes.h>


#define N 8
#define N2 64 // N*N
#define M 6
#define M2 36

void forward(float (&arg_2)[1][1][N][N], float (&arg_3)[1][1][M][M]);

#define DWIDTH 512
typedef ap_axiu<DWIDTH, 0, 0, 0> axis_t;

typedef ap_uint<512> uint512_t;
typedef float DataType;

const int DataTypeSize = sizeof(DataType) * 8;

typedef ap_uint<DataTypeSize> DataTypeInt;

typedef union converter {
  DataType d;
  uint32_t i;
} converter_t;

/***************************************************************************
***************************************************************************/
#include "hls_stream.h"

extern "C" {
void wrapper(hls::stream<axis_t> &in, hls::stream<axis_t> &out) {
//#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS INTERFACE axis port = in
#pragma HLS INTERFACE axis port = out

  float l_A[1][1][8][8];
  float l_C[1][1][6][6];

  //#pragma HLS ARRAY_PARTITION variable = l_A factor = 16 dim = 1 cyclic
  //#pragma HLS ARRAY_PARTITION variable = l_C factor = 16 dim = 1 cyclic

  int j_limit = 512 / DataTypeSize; // 512 / 4 = 128
  int i_limit = N2 / j_limit;
  converter_t converter;

load_A:
  for (int i = 0; i < i_limit; i++) {
    axis_t temp = in.read();
    for (int j = 0; j < j_limit; j++) {
      int high = j * DataTypeSize + DataTypeSize - 1;
      int low = j * DataTypeSize;

      converter.i = temp.data.range(high, low);
      l_A[1][1][i][j] = converter.d;
    }
  }

  forward(l_A, l_C);

  int k_limit = M2 / j_limit;
writeC:
  for (int k = 0; k < k_limit; k++) {
    axis_t temp;
    for (int j = 0; j < j_limit; j++) {
      int high = j * DataTypeSize + DataTypeSize - 1;
      int low = j * DataTypeSize;
      converter.d = l_C[0][0][k][j];
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

