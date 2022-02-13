#ifndef CTC_DECODER_LAYER_H
#define CTC_DECODER_LAYER_H
#include "ctc_decode_util.h"
#include <string>
#include <vector>
#include <utility>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <map>

// beam 束搜索
std::string ctc_beam_search_decoder(
    std::vector<std::vector<float> > probs_seq,
    int beam_size,
    std::vector<std::string> vocabulary,
    int blank_id,
    float cutoff_prob);

// 贪心搜索
std::string ctc_greedy_decoder(
    const std::vector<std::vector<float>> &probs_seq,
    const std::vector<std::string> &vocabulary);
// #ifdef __cplusplus
// }
// #endif
#endif //CTC_DECODER_LAYER_H