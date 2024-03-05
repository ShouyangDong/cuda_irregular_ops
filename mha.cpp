void multiHeadAttentionForward(
    float Q, //DIM: [Q_LEN][DIM], 
    float K, //DIM: [K_LEN][DIM], 
    float V, //DIM: [K_LEN][DIM], 
    float output, //DIM: [Q_LEN][DIM]
){
    float scale = 1.0 / sqrt((double) DIM * 1.0 / HEAD_SIZE);
    float tmp[HEAD_SIZE][Q_LEN][DIM];
    for(int i = 0; i < HEAD_SIZE; ++i){
        float q_tmp_1[Q_LEN][DIM];
        float q_tmp[Q_LEN][DIM];
        float k_tmp[K_LEN][DIM];
        float v_tmp[K_LEN][DIM];
        // linearForward(Q, q_tmp_1, param.q_param);
        for(int i_qp = 0; i_qp < Q_LEN; ++i_qp){
            for(int i_bias_q = 0; i_bias_q < DIM; ++i_bias_q){
                q_tmp_1[i_qp][i_bias_q] = q_param_bias[i_qp][i_bias_q];
            }
            for(int i_q = 0; i_q < DIM; ++i_q){
                for(int j_qp = 0; j_qp < DIM; ++j_qp){
                    q_tmp_1[i_qp][j_qp] += Q[i_qp][i_q] * q_param_weights[i_q][j_qp];
                }
            }
        }

        // linearForward(K, k_tmp, param.k_param);
        for(int i_kp = 0; i_kp < K_LEN; ++i_kp){
            for(int i_bias_k = 0; i_bias_k < DIM; ++i_bias_k){
                k_tmp[i_kp][i_bias_k] = k_param_bias[i_kp][i_bias_k];
            }
            for(int i_k = 0; i_k < DIM; ++i_k){
                for(int j_kp = 0; j_kp < DIM; ++j_kp){
                    k_tmp[i_kp][j_kp] += K[i_kp][i_k] * k_param_weights[i_k][j_kp];
                }
            }
        }


        // linearForward(V, v_tmp, param.v_param);
        for(int i_vp = 0; i_vp < K_LEN; ++i_vp){
            for(int i_bias_v = 0; i_bias_v < DIM; ++i_bias_v){
                v_tmp[i_vp][i_bias_v] = v_param_bias[i_vp][i_bias_v];
            }
            for(int i_v = 0; i_v < DIM; ++i_v){
                for(int j_vp = 0; j_vp < DIM; ++j_vp){
                    v_tmp[i_vp][j_vp] += V[i_vp][i_v] * v_param_weights[i_v][j_vp];
                }
            }
        }


        for(int i_dp_h = 0; i_dp_h < Q_LEN; ++i_dp_h){
            // dropoutForward<T, DIM>(q_tmp_1[i], q_tmp[i], dr);
            for (int i_dp = 0; i_dp < DIM; ++i_dp) {
                if (q_tmp_1[i_dp_h][i_dp] < dropout_rate) {
                    q_tmp[i_dp_h][i_dp] = 0;
                } else {
                    q_tmp[i_dp_h][i_dp] = q_tmp_1[i_dp_h][i_dp];
                }
            }

            for(int j_scale = 0; j_scale < DIM; ++j_scale){
                q_tmp[i_dp_h][j_scale] *= scale;
            }
        }


        float nex_tmp[Q_LEN][K_LEN];
        for(int i_qk = 0; i_qk < Q_LEN; ++i_qk){
            for(int j_qk = 0; j_qk < K_LEN; ++j_qk){
                nex_tmp[i_qk][j_qk] = 0;
                for(int k_qk = 0; k_qk < DIM; ++k_qk){
                    nex_tmp[i_qk][j_qk] += q_tmp[i_qk][k_qk] * k_tmp[j_qk][k_qk];
                }
            }
        }
        float nex_tmp_2[Q_LEN][K_LEN];
        // softmaxForward<T, K_LEN, Q_LEN>(nex_tmp, nex_tmp_2);
        // The Softmax code:
        for (int j_sf = 0; j_sf < Q_LEN; ++j_sf) {
            float sum = 0;
            for (int i_sf = 0; i_sf < K_LEN; ++i_sf) {
                sum += nex_tmp[j_sf][i_sf];
            }
            for (int k_sf = 0; k_sf < K_LEN; ++k_sf) {
                nex_tmp_2[j_sf][k_sf] = nex_tmp[j_sf][k_sf] / sum;
            }
        }

        for(int i_dot_v = 0; i_dot_v < Q_LEN; ++i_dot_v){
            for(int j_dot_v = 0; j_dot_v < DIM; ++j_dot_v){
                tmp[i][i_dot_v][j_dot_v] = 0;
                for(int k_dot_v = 0; k_dot_v < K_LEN; ++k_dot_v){
                    tmp[i][i_dot_v][j_dot_v] += nex_tmp_2[i_dot_v][k_dot_v] * v_tmp[k_dot_v][j_dot_v];
                }
            }
        }
    }
    float fc_tmp[Q_LEN][DIM * HEAD_SIZE];
    for(int h = 0; h < HEAD_SIZE; ++h){
        for(int i_tp = 0; i_tp < Q_LEN; ++i_tp){
            for(int j_tp = 0; j_tp < DIM; ++j_tp){
                fc_tmp[i_tp][h*HEAD_SIZE+j_tp] = tmp[h][i_tp][j_tp];
            }
        }
    }
    // linearForward<T, DIM * HEAD_SIZE, DIM, Q_LEN>(fc_tmp, output, param.lp);
    for(int i_lp = 0; i_lp < Q_LEN; ++i_lp){
        for(int j_lp = 0; j_lp < DIM; ++j_lp){
            output[i_lp][j_lp] = lp_param_bias[i_lp][j_lp];
        }
        for(int i_in = 0; i_in < DIM * HEAD_SIZE; ++i_in){
            for(int j_out = 0; j_out < DIM; ++j_out){
                output[j_out] += fc_tmp[i_in] * lp_param_weights[i_in][j_out];
            }
        }

    }
}