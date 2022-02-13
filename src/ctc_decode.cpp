#include "ctc_decode.h"

std::string ctc_greedy_decoder(
    const std::vector<std::vector<float>> &probs_seq,
    const std::vector<std::string> &vocabulary) {
  // dimension check
  size_t num_time_steps = probs_seq.size();
  for (size_t i = 0; i < num_time_steps; ++i) {
    VALID_CHECK_EQ(probs_seq[i].size(),
                   vocabulary.size() + 1,
                   "The shape of probs_seq does not match with "
                   "the shape of the vocabulary");
  }

  size_t blank_id = vocabulary.size();

  std::vector<size_t> max_idx_vec(num_time_steps, 0);
  std::vector<size_t> idx_vec;
  for (size_t i = 0; i < num_time_steps; ++i) {
    float max_prob = 0.0;
    size_t max_idx = 0;
    const std::vector<float> &probs_step = probs_seq[i];
    for (size_t j = 0; j < probs_step.size(); ++j) {
      if (max_prob < probs_step[j]) {
        max_idx = j;
        max_prob = probs_step[j];
      }
    }
    // id with maximum probability in current time step
    max_idx_vec[i] = max_idx;
    // deduplicate
    if ((i == 0) || ((i > 0) && max_idx_vec[i] != max_idx_vec[i - 1])) {
      idx_vec.push_back(max_idx_vec[i]);
    }
  }

  std::string best_path_result;
  for (size_t i = 0; i < idx_vec.size(); ++i) {
    if (idx_vec[i] != blank_id) {
      best_path_result += vocabulary[idx_vec[i]];
    }
  }
  return best_path_result;
}

std::string ctc_beam_search_decoder(std::vector<std::vector<float> > probs_seq,
                                   int beam_size,
                                   std::vector<std::string> vocabulary,
                                   int blank_id,
                                   float cutoff_prob) {
    int num_time_steps = probs_seq.size();
    
    // assign space ID 
    //std::vector<std::string>::iterator it = std::find(vocabulary.begin(), vocabulary.end(), " ");
    
    // initialize
    // two sets containing selected and candidate prefixes respectively
    std::map<std::string, float> prefix_set_prev, prefix_set_next;
    // probability of prefixes ending with blank and non-blank
    std::map<std::string, float> probs_b_prev, probs_nb_prev;
    std::map<std::string, float> probs_b_cur, probs_nb_cur;
    prefix_set_prev[" "] = 0;
    probs_b_prev[" "] = 0;
    probs_nb_prev[" "] = -NUM_FLT_INF;
    
    for (int time_step=0; time_step<num_time_steps; time_step++) 
		{
	//			std::cout<<"The "<<time_step<<" frame is decoding"<<std::endl;
        prefix_set_next.clear();
        probs_b_cur.clear();
        probs_nb_cur.clear();
        std::vector<float> prob = probs_seq[time_step];

        std::vector<std::pair<int, float> > prob_idx;
				std::vector<std::pair<int, float>> log_prob_idx;
        for (int i = 0; i < prob.size(); i++) 
				{
              prob_idx.push_back(std::pair<int,float>(i, prob[i]));
        }
        // pruning of vacobulary
        std::sort(prob_idx.begin(), prob_idx.end(), pair_comp_second_rev<int, float>);
        if (cutoff_prob < 1.0)
				{
					 float cum_prob = 0.0;
           int cutoff_len = 0;
           for (int i=0; i < prob_idx.size(); i++) 
					 {
             //cum_prob += std::exp(prob_idx[i].second);
              cum_prob += prob_idx[i].second;
              cutoff_len += 1;
              if (cum_prob >= cutoff_prob) break;
           }
           prob_idx = std::vector<std::pair<int, float> >(prob_idx.begin(), prob_idx.begin()+cutoff_len);
					 for (int i = 0; i < cutoff_len; ++i)
						 log_prob_idx.push_back(std::pair<int, float>(prob_idx[i].first, std::log(prob_idx[i].second + NUM_FLT_MIN)));
        }
        // extend prefix
        for (std::map<std::string, float>::iterator it = prefix_set_prev.begin(); 
            it != prefix_set_prev.end(); it++) 
						{
            std::string l = it->first;
//						std::cout<<l<<"\n";
            if( prefix_set_next.find(l) == prefix_set_next.end()) 
						{
                probs_b_cur[l] = probs_nb_cur[l] = -NUM_FLT_INF;
            }

            for (int index=0; index<log_prob_idx.size(); index++) 
						{
                int c = log_prob_idx[index].first;
                float prob_c = log_prob_idx[index].second;
                if (c == blank_id) {
                    probs_b_cur[l] = log_sum_exp(probs_b_cur[l],(prob_c+(log_sum_exp(probs_b_prev[l],probs_nb_prev[l]))));
                } else {
                    std::string last_char = l.substr(l.size()-1, 1);
                    std::string new_char = vocabulary[c];
                    std::string l_plus = l+" "+new_char;

                    if( prefix_set_next.find(l_plus) == prefix_set_next.end()) 
										{
                        probs_b_cur[l_plus] = probs_nb_cur[l_plus] = -NUM_FLT_INF;
                    }

                    if (last_char == new_char) 
										{
                        probs_nb_cur[l_plus] =log_sum_exp(probs_nb_cur[l_plus],(prob_c + probs_b_prev[l]));
                        probs_nb_cur[l] =log_sum_exp(probs_nb_cur[l], (prob_c +probs_nb_prev[l]));
                    }
										
                    prefix_set_next[l_plus] = log_sum_exp(probs_nb_cur[l_plus],probs_b_cur[l_plus]);
                }
							
            }

            prefix_set_next[l] = log_sum_exp(probs_b_cur[l],probs_nb_cur[l]);  
        }

        probs_b_prev = probs_b_cur;
        probs_nb_prev = probs_nb_cur;
        std::vector<std::pair<std::string, float> > 
                  prefix_vec_next(prefix_set_next.begin(), prefix_set_next.end());
        std::sort(prefix_vec_next.begin(), prefix_vec_next.end(), pair_comp_second_rev<std::string, float>);
        int k = beam_size<prefix_vec_next.size() ? beam_size : prefix_vec_next.size();
        prefix_set_prev = std::map<std::string, float>
                  (prefix_vec_next.begin(), prefix_vec_next.begin()+k);
    }
    // post processing
    std::vector<std::pair<float, std::string> > beam_result;
    for (std::map<std::string, float>::iterator it = prefix_set_prev.begin(); 
         it != prefix_set_prev.end(); it++) 
		{
        if (it->second > -NUM_FLT_INF && it->first.size() > 1) 
				{
            float prob = it->second;
            std::string sentence = it->first.substr(1);
            float log_prob = prob;
            beam_result.push_back(std::pair<float, std::string>(log_prob, sentence));
        }
    }
    // sort the result and return
    std::sort(beam_result.begin(), beam_result.end(), pair_comp_first_rev<float, std::string>);
    std::string search_result;
    search_result = beam_result[0].second;
    return search_result;
}
