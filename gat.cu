#include <gpu_runtime.h>
extern __shared__ uint8_t __shmem[];

extern "C" {
__global__ void __launch_bounds__(32 * 2 * 1) _inference_kernel0(mdspan_r<float, extents<4267, 32>> _Y, mdspan_r<float, extents<4267, 32>> _y, mdspan_r<const int32_t, extents<2135822>> _idx, mdspan_r<float, extents<4267>> _Y__1, mdspan_r<float, extents<67, 2, 32, 2135822>> _edge__exp, mdspan_r<float, extents<67, 2, 32, 2135822>> _edge, mdspan_r<float, extents<4267>> _Y__2, mdspan_r<const int32_t, extents<4268>> _ptr, __ByValArray<void *, 7> params, uint8_t *__glmem) {
   if (((int)threadIdx.x < (((((int)threadIdx.y == 1) && ((int)blockIdx.x == 66)) ? 10 : 31) + 1))) {
     float _edge__max;
     _edge__max = -INFINITY;
     for (int _k = _ptr((((int)threadIdx.x + (64 * (int)blockIdx.x)) + (32 * (int)threadIdx.y))); _k < _ptr(((((int)threadIdx.x + (64 * (int)blockIdx.x)) + (32 * (int)threadIdx.y)) + 1)); _k++) {
       _edge((int)blockIdx.x, (int)threadIdx.y, (int)threadIdx.x, _k) = (((_Y__1(_idx(_k)) + _Y__2((((int)threadIdx.x + (64 * (int)blockIdx.x)) + (32 * (int)threadIdx.y)))) >= 0) ? (_Y__1(_idx(_k)) + _Y__2((((int)threadIdx.x + (64 * (int)blockIdx.x)) + (32 * (int)threadIdx.y)))) : ((_Y__1(_idx(_k)) + _Y__2((((int)threadIdx.x + (64 * (int)blockIdx.x)) + (32 * (int)threadIdx.y)))) * 0x1.99999ap-4f));
       _edge__max = max(_edge__max, _edge((int)blockIdx.x, (int)threadIdx.y, (int)threadIdx.x, _k));
     }
     {
       float _edge__sum;
       _edge__sum = 0;
       for (int _k__1 = _ptr((((int)threadIdx.x + (64 * (int)blockIdx.x)) + (32 * (int)threadIdx.y))); _k__1 < _ptr(((((int)threadIdx.x + (64 * (int)blockIdx.x)) + (32 * (int)threadIdx.y)) + 1)); _k__1++) {
         _edge__exp((int)blockIdx.x, (int)threadIdx.y, (int)threadIdx.x, _k__1) = runtime_exp((_edge((int)blockIdx.x, (int)threadIdx.y, (int)threadIdx.x, _k__1) - _edge__max));
         _edge__sum += _edge__exp((int)blockIdx.x, (int)threadIdx.y, (int)threadIdx.x, _k__1);
       }
       for (int _fuse__i0 = 0; _fuse__i0 < 32; _fuse__i0++) {
         if ((((-1 * _ptr((((int)threadIdx.x + (64 * (int)blockIdx.x)) + (32 * (int)threadIdx.y)))) + _ptr(((((int)threadIdx.x + (64 * (int)blockIdx.x)) + (32 * (int)threadIdx.y)) + 1))) > 0)) {
           _y((((int)threadIdx.x + (64 * (int)blockIdx.x)) + (32 * (int)threadIdx.y)), _fuse__i0) = 0;
         }
         for (int _rem1__i1 = 0; _rem1__i1 < min(2147483647, ((-1 * _ptr((((int)threadIdx.x + (64 * (int)blockIdx.x)) + (32 * (int)threadIdx.y)))) + _ptr(((((int)threadIdx.x + (64 * (int)blockIdx.x)) + (32 * (int)threadIdx.y)) + 1)))); _rem1__i1++) {
           _y((((int)threadIdx.x + (64 * (int)blockIdx.x)) + (32 * (int)threadIdx.y)), _fuse__i0) += ((_Y(_idx((_rem1__i1 + _ptr((((int)threadIdx.x + (64 * (int)blockIdx.x)) + (32 * (int)threadIdx.y))))), _fuse__i0) * _edge__exp((int)blockIdx.x, (int)threadIdx.y, (int)threadIdx.x, (_rem1__i1 + _ptr((((int)threadIdx.x + (64 * (int)blockIdx.x)) + (32 * (int)threadIdx.y)))))) / _edge__sum);
         }
       }
     }
   }
}
}
