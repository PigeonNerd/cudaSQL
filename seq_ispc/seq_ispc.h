#ifndef ISPC_SEQ_ISPC_H
#define ISPC_SEQ_ISPC_H
#include <stdint.h>

#ifdef __cplusplus
namespace ispc {
#endif

#if defined(__cplusplus) && !defined(__ISPC_NO_EXTERN_C)
 extern "C" {
#endif	
 	void mandelbrot_ispc_withtasks( int32_t N, int32_t input[], int32_t output[]);
#if defined(__cplusplus) && !defined(__ISPC_NO_EXTERN_C)    
}
#endif

#ifdef __cplusplus
}
#endif

#endif
