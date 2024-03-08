extern "C" void multiHeadAttentionForward_kernel(
    float Q, //[batch, seq_len, heads, dim] 
    float K, //[batch, seq_len, heads, dim]
    float V, //[batch, seq_len, heads, dim]
    float output, //[batch, seq_len, heads, dim]
){
    // The dimension 64, 2048, 12, 256
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 2048; j++) {
            for (int m = 0; m < 12; m++) {
                for (int n = 0; n < 12; n++) {
                    result[i][j][m][n] = 0.0;
                    for (int p = 0; p < 256; p++) {
                        result[i][j][m][n] += q[i][j][m][p] * k[i][j][n][p];
                    }
                }
            }

            // score 
            for (int m_sc = 0; m_sc < 12; m_sc++) {
                for (int n_sc = 0; n_sc < 12; n_sc++) {
                    score[i][j][m][n] = score[i][j][m][n] / 256;
                }
            }

            // The Softmax code:
            for (int j_sf = 0; j_sf < 12; ++j_sf) {
                float sum = 0;
                for (int i_sf = 0; i_sf < 12; ++i_sf) {
                    sum += score[i][j][j_sf][i_sf];
                }
                for (int k_sf = 0; k_sf < 12; ++k_sf) {
                    score[j_sf][i][j][k_sf] = score[i][j][j_sf][k_sf] / sum;
                }
            }


            // The final Matmul
            for (int m_fl = 0; m_fl < 12; ++m_fl) {
                for (int n_fl = 0; n_fl < 256; ++n_fl) {
                    output[i][j][m_fl][n_fl] = 0.0;
                    for (int k_fl = 0; k_fl < 12; ++k_fl) {
                        output[i][j][m_fl][k_fl] += score[i][j][m_fl][k_fl] * V[i][j][k_fl][n_fl];
                    }
                }

            }

        }
    }
}