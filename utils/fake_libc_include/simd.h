/* stub out the SIMD type and intrinsics */
typedef struct { long long v[2]; } __m128i;

__m128i _mm_setzero_si128(void);
__m128i _mm_loadu_si128(const __m128i *p);
__m128i _mm_dpbusds_epi32(__m128i a, __m128i b, __m128i c);
void     _mm_storeu_si128(__m128i *p, __m128i v);
