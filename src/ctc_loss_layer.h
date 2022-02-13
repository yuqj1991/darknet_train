#ifndef CTC_LOSS_LAYER_H
#define CTC_LOSS_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"
#include <string>
#include <vector>

layer make_ctc_loss_layer(int batch, int time_steps, int blank_label, int alphabet_size);
void forward_ctc_loss_layer(const layer l, network_state state);
void backward_ctc_loss_layer(const layer l, network_state state);
void resize_ctc_loss_layer(layer *l, int w, int h);
int ctc_num_detections(layer l, float thresh);
#ifdef GPU
void forward_ctc_loss_layer_gpu(const layer l, network_state state);
void backward_ctc_loss_layer_gpu(layer l, network_state state);
#endif

CTC_Result_S* get_ctc_result(layer l, CTC_DECODE_METHOD decode_method,
                           const char** label_vocabulary, const int voc_length);

#endif //CTC_LOSS_LAYER_H