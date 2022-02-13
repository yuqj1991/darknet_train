//
// Created by yuqianjin
//

//
// Created by yuqianjin 
//

#include "polygon.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "dark_cuda.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>


layer make_polygon_layer(int batch, int classes)
{
    layer l = { (LAYER_TYPE)0 };
    l.type = POLYGON;
    l.batch = batch;
    l.classes = classes;
    l.cost = (float*)xcalloc(1, sizeof(float));
    l.outputs = (8 + 1) * classes;
    l.inputs = l.outputs;
    l.truths = (8 + 1) * classes;
    l.delta = (float*)xcalloc(batch*l.outputs, sizeof(float));
    l.output = (float*)xcalloc(batch*l.outputs, sizeof(float));
    
    l.forward = forward_polygon_layer;
    l.backward = backward_polygon_layer;

#ifdef GPU
    l.forward_gpu = forward_polygon_layer_gpu;
    l.backward_gpu = backward_polygon_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);

    free(l.output);
    if (cudaSuccess == cudaHostAlloc(&l.output, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.output_pinned = 1;
    else { 
        cudaGetLastError(); // reset CUDA-error
        l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
    }

    free(l.delta);
    if (cudaSuccess == cudaHostAlloc(&l.delta, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.delta_pinned = 1;
    else { 
        cudaGetLastError(); // reset CUDA-error
        l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
    }
#endif
    
    fprintf(stderr, "polygon ");
    srand(time(0));
    return l;    
}

float sigmoid_loss (float label_id, float pred, float *delta) {
    float loss = 0.;
    if (label_id == -1) {
        loss = -log(1 - pred);
        *delta = pred;
    } else {
        loss = -log(pred);
        *delta = pred - 1;
    }
    return loss;

}

float smoothL1_Loss_poly(float x, float* x_diff){
    float loss = 0.;
    float fabs_x_value = fabs(x);
    if(fabs_x_value < 1){
        loss = 0.5 * x * x;
        *x_diff = x;
    }else{
        loss = fabs_x_value - 0.5;
        *x_diff = (0. < x) - (x < 0.);
    }
    return loss;
}

float delta_polygon_box(PolyGon_S* truth, float *x, float *box_loss, float* delta)
{
    float temp_loss = 0, delta_x1 = 0, delta_y1 = 0, delta_x2 = 0, delta_y2 = 0;
    float delta_x3 = 0,  delta_y3 = 0, delta_x4 = 0, delta_y4 = 0;
    for (auto i = 0; i < 2; i++) {
        if (truth[i].label_id == -1)
            continue;
        temp_loss += smoothL1_Loss_poly(x[i * 9 + 1] - truth[i].leftTop.x, &delta_x1);
        temp_loss += smoothL1_Loss_poly(x[i * 9 + 2] - truth[i].leftTop.x, &delta_y1);
        temp_loss += smoothL1_Loss_poly(x[i * 9 + 3] - truth[i].rightTop.x, &delta_x2);
        temp_loss += smoothL1_Loss_poly(x[i * 9 + 4] - truth[i].rightTop.x, &delta_y2);
        temp_loss += smoothL1_Loss_poly(x[i * 9 + 5] - truth[i].rightBottom.x, &delta_x3);
        temp_loss += smoothL1_Loss_poly(x[i * 9 + 6] - truth[i].rightBottom.x, &delta_y3);
        temp_loss += smoothL1_Loss_poly(x[i * 9 + 7] - truth[i].leftBottom.x, &delta_x4);
        temp_loss += smoothL1_Loss_poly(x[i * 9 + 8] - truth[i].leftBottom.x, &delta_y4);
        delta[i * 9 + 1] = delta_x1 * (-1) * x[i * 9 + 1] * (1 - x[i * 9 + 1]);
        delta[i * 9 + 2] = delta_y1 * (-1) * x[i * 9 + 2] * (1 - x[i * 9 + 2]);
        delta[i * 9 + 3] = delta_x2 * (-1) * x[i * 9 + 3] * (1 - x[i * 9 + 3]);
        delta[i * 9 + 4] = delta_y2 * (-1) * x[i * 9 + 4] * (1 - x[i * 9 + 4]);
        delta[i * 9 + 5] = delta_x3 * (-1) * x[i * 9 + 5] * (1 - x[i * 9 + 5]);
        delta[i * 9 + 6] = delta_y3 * (-1) * x[i * 9 + 6] * (1 - x[i * 9 + 6]);
        delta[i * 9 + 7] = delta_x4 * (-1) * x[i * 9 + 7] * (1 - x[i * 9 + 7]);
        delta[i * 9 + 8] = delta_y4 * (-1) * x[i * 9 + 8] * (1 - x[i * 9 + 8]);
    }
    
    
    *box_loss = temp_loss;
    return temp_loss;
}

void forward_polygon_layer(const layer l, network_state state)
{
    int i,j,b;
    float loss;
    memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
    #ifndef GPU
    for (b = 0; b < l.batch; ++b){
        activate_array(l.output + b * l.outputs, l.outputs, LOGISTIC);
    }
    #endif
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!state.train){
        return;
    }
    float avg_iou = 0;
    float avg_obj = 0;
    int count = 0;
    *(l.cost) = 0;
    float alpha = 2., gamma_ = 4.;
    float class_cost=0, box_cost = 0;

    PolyGon_S* poly_label = (PolyGon_S*) xcalloc(2, sizeof(PolyGon_S));
    float batch_loss = 0.;
    for (b = 0; b < l.batch; ++b) {
        convertToPoly(state.truth + b * 18, poly_label);
        // 1)计算 左页 右页的label id 损失，使用sigmoid，损失函数吗？
        float sigmoid_loss_vale = 0.;
        sigmoid_loss_vale = sigmoid_loss(poly_label[0].label_id, l.output[b * l.outputs + 0], &(l.delta[b * l.outputs + 0]));
        sigmoid_loss_vale += sigmoid_loss(poly_label[0].label_id, l.output[b * l.outputs + 9], &(l.delta[b * l.outputs + 9]));
        // 2) 计算每一页的坐标损失，使用smoothL1
        float poly_loss = 0.;
        delta_polygon_box(poly_label, l.output + b * l.outputs, &poly_loss, l.delta + b * l.outputs);
        batch_loss += (poly_loss + sigmoid_loss_vale);
    }
    *(l.cost) = batch_loss / l.batch;
    printf("PolyGon loss : %f \n", *(l.cost));
}

void backward_polygon_layer(const layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1 / l.batch, l.delta, 1, state.delta, 1);
}



int get_poly_result(float *output, PolyGon_S* poly_result, float predict_threhold)
{
    int count = 0;
    convertToPoly(output, poly_result);
    for (auto i = 0; i < 2; i++) {
        if (poly_result[i].label_id >= predict_threhold) {
            count ++;
        } else {
            poly_result[i].label_id = -1;
        }
    }
    return count;
}

#ifdef GPU

void forward_polygon_layer_gpu(const layer l, network_state state)
{
    simple_copy_ongpu(l.batch*l.inputs, state.input, l.output_gpu);
    for (int b = 0; b < l.batch; ++b){
        activate_array_ongpu(l.output_gpu + b * l.outputs, l.outputs, LOGISTIC);
    }
    if(!state.train || l.onlyforward) {
        cuda_pull_array_async(l.output_gpu, l.output, l.batch*l.outputs);
        CHECK_CUDA(cudaPeekAtLastError());
        return;
    }
    float *in_cpu = (float *)xcalloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    memcpy(in_cpu, l.output, l.batch*l.outputs*sizeof(float));
    float *truth_cpu = 0;
    
    if (state.truth) {
        int num_truth = l.batch*l.truths;
        truth_cpu = (float *)xcalloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    network_state cpu_state = state;
    cpu_state.net = state.net;
    cpu_state.index = state.index;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_polygon_layer(l, cpu_state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    free(in_cpu);
    if (cpu_state.truth) free(cpu_state.truth);
}

void backward_polygon_layer_gpu(const layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1 / l.batch, l.delta_gpu, 1, state.delta, 1);
}
#endif
