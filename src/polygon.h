#ifndef POLYGON_LAYER_H
#define POLYGON_LAYER_H
//#include "darknet.h"
#include "layer.h"
#include "network.h"
#ifdef __cplusplus
extern "C" {
#endif
layer make_polygon_layer(int batch, int classes);
void forward_polygon_layer(const layer l, network_state state);
void backward_polygon_layer(const layer l, network_state state);
int get_poly_result(float *output, PolyGon_S* poly_result, float predict_threhold);
#ifdef GPU
void forward_polygon_layer_gpu(const layer l, network_state state);
void backward_polygon_layer_gpu(layer l, network_state state);
#endif
#ifdef __cplusplus
}
#endif
#endif //POLYGON_LAYER_H
