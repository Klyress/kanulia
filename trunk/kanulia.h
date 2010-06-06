#ifndef _MANDELBROT_KERNAL_h_
#define _MANDELBROT_KERNAL_h_

#include <vector_types.h>

// 4D rotations angles for Julia 4D
__device__ __constant__ float aanglexw;
__device__ __constant__ float aangleyw;
__device__ __constant__ float aanglexy;
__device__ __constant__ float aanglexz;

// 4D position of Julia4D seed point
__device__ __constant__ double xJS;
__device__ __constant__ double yJS;
__device__ __constant__ double zJS;
__device__ __constant__ double wJS;

__device__ __constant__ int crn; // Max itération de la Julia
__device__ __constant__ int numSM; // KO number of multiprocessors on device
__device__ __constant__ int imgW; // KO image Width
__device__ __constant__ int imgH; // KO image Height

extern "C" void RunJulia4Drepart(uchar4 *dst, const int imageW, const int imageH,
					const double xOff, const double yOff, const double zOff, const double wOff,
					const double scale, const double scalei,
					const double xJOff, const double yJOff, const double scaleJ,
					const float xblur, const float yblur, // blur coeff for julia 4D
					const uchar4 colors, const int frame, const int animationFrame, const int mode, const int numSMs, const int julia, const int julia4D);

__device__ inline void HSL2RGB(float h, const float sl, const float ll, int *rc, int *gc, int *bc);

template<class T>
__device__ inline int CalcJulia4Dhue(const T xPos, const T yPos, const T zPos, const T wPos, float *hue);
template<class T>
__device__ inline int CalcMandel4Dcore(const T xPos, const T yPos, const T zPos, const T wPos, float *hue);

					
extern "C" int inEmulationMode();

template<class T> __device__ inline void rotate4(T *px, T *py, T *pz, T *pw);

#endif
