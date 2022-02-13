#include "ctc_loss_layer.h"
#include "ctc_decode.h"
#include "utils.h"
#include "ctc.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

inline void throw_on_error(ctcStatus_t status, const char* message) {
    if (status != CTC_STATUS_SUCCESS) {
        error(message, DARKNET_LOC);
    }
}

layer make_ctc_loss_layer(int batch, int time_steps, int blank_label, int alphabet_size) {
    layer l = { (LAYER_TYPE)0 };
    l.type = CTC_LOSS;
    l.batch = batch;

    l.time_steps = time_steps;
    l.alphabet_size = alphabet_size;
    l.blank_index = blank_label;

    l.cost = (float*)xcalloc(batch, sizeof(float));
    l.outputs = l.time_steps * l.alphabet_size;
    l.inputs = l.outputs;

    l.input_length = (int*)xcalloc(l.batch, sizeof(int));
    for (auto i = 0; i < l.batch ; i++) {
        l.input_length[i] = time_steps;
    }

    l.delta = (float*)xcalloc(batch*l.outputs, sizeof(float));
    l.output = (float*)xcalloc(batch*l.outputs, sizeof(float));

    l.forward = forward_ctc_loss_layer;
    l.backward = backward_ctc_loss_layer;

#ifdef GPU
    l.forward_gpu = forward_ctc_loss_layer_gpu;
    l.backward_gpu = backward_ctc_loss_layer_gpu;
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
    
    fprintf(stderr, "ctdet ");
    srand(time(0));

    return l;
    
}

void forward_ctc_loss_layer(const layer l, network_state state) {
    ctcOptions options;
    options.loc = CTC_CPU;
    options.num_threads = 1;
    options.blank_label = l.blank_index;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(state.ocr_label.label_length,
                                      l.input_length,
                                      l.alphabet_size,
                                      l.batch,
                                      options,
                                      &cpu_alloc_bytes),
                                  "Error: get_workspace_size in options_test");

    void* ctc_cpu_workspace = malloc(cpu_alloc_bytes);
    std::vector<int> ocr_labels;
    for (auto i = 0; i < l.batch; i++) {
        for (auto j = 0; j < state.ocr_label.label_length[i]; j++) {
            ocr_labels.push_back(state.ocr_label.labels[i][j]);
        }
    }

    throw_on_error(compute_ctc_loss(state.input,
                                    l.delta,
                                    ocr_labels.data(),
                                    state.ocr_label.label_length,
                                    l.input_length,
                                    l.alphabet_size,
                                    l.batch,
                                    l.cost,
                                    ctc_cpu_workspace,
                                    options),
                                    "Error: compute_ctc_loss in options_test");

    float cost_loss = 0.f;
    for (auto i = 0; i < l.batch; i++) {
        cost_loss += l.cost[i];
    }
    cost_loss /= l.batch;
    printf("ctc_loss: %f", cost_loss);
}

void backward_ctc_loss_layer(const layer l, network_state state) {
    ;
}

void resize_ctc_loss_layer(layer *l, int w, int h) {
    ;
}

#ifdef GPU
void forward_ctc_loss_layer_gpu(layer l, network_state state) {
    ctcOptions options;
    options.loc = CTC_GPU;
    options.num_threads = 1;
    options.blank_label = l.blank_index;
    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(state.ocr_label.label_length,
                                      l.input_length,
                                      l.alphabet_size,
                                      l.batch, options,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in options_test");

    void* ctc_gpu_workspace = malloc(gpu_alloc_bytes);

    float* cost_batch = (float*)xcalloc(l.batch, sizeof(float));

    std::vector<int> ocr_labels;
    for (auto i = 0; i < l.batch; i++) {
        for (auto j = 0; j < state.ocr_label.label_length[i]; j++) {
            ocr_labels.push_back(state.ocr_label.labels[i][j]);
        }
    }

    throw_on_error(compute_ctc_loss(state.input,
                                    l.delta_gpu,
                                    ocr_labels.data(),
                                    state.ocr_label.label_length,
                                    l.input_length,
                                    l.alphabet_size,
                                    l.batch,
                                    l.cost,
                                    ctc_gpu_workspace,
                                    options),
                   "Error: compute_ctc_loss in options_test");
    float cost_loss = 0.f;
    for (auto i = 0; i < l.batch; i++) {
        cost_loss += l.cost[i];
    }
    cost_loss /= l.batch;
    printf("ctc_loss: %f", cost_loss);
}

void backward_ctc_loss_layer_gpu(const layer l, network_state state) {
    ;
}
#endif

CTC_Result_S* get_ctc_result(layer l, CTC_DECODE_METHOD decode_method,
                             const char** label_vocabulary, const int voc_length) {
    CTC_Result_S* ctc_result = (CTC_Result_S*)xcalloc(l.batch, sizeof(CTC_Result_S));
    std::vector<std::string> vocabulary;
    for (auto i = 0; i < voc_length; i++) {
        vocabulary.push_back(label_vocabulary[i]);
    }
    for (auto i = 0; i < l.batch; i++) {
        std::string result_vec;
        std::vector<std::vector<float>> probs;
        for (int time_step = 0; time_step < l.time_steps; time_step++) {
            std::vector<float> time_probs;
            for (int j = 0; j < voc_length; j++) {
                time_probs.push_back(l.output[time_step * l.batch * l.alphabet_size + i * l.alphabet_size + j]);
            }
            probs.push_back(time_probs);
        }
        if (decode_method == GREEDY_SEARCH) {
            result_vec = ctc_greedy_decoder(probs, vocabulary);
        } else if (decode_method == BEAM_SEARCH) {
            int blank_label = voc_length;
            float cutoff_prob = 0.95;
            result_vec = ctc_beam_search_decoder(probs, 5, vocabulary, blank_label,
                                                cutoff_prob);
        }
        ctc_result[0].time_steps = result_vec.length();
        ctc_result[0].ocr_result = (char*)xcalloc(ctc_result[0].time_steps,
                                                                sizeof(char));
        memcpy(ctc_result[0].ocr_result, result_vec.c_str(),
                                                    ctc_result[0].time_steps);
    }
    return ctc_result;
}
