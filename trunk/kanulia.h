#ifndef _MANDELBROT_KERNAL_h_
#define _MANDELBROT_KERNAL_h_

#include <vector_types.h>

#define OUTMANDELBOX 4.0
#define RAYSTEP 0.0007

// Flag for Julia type
#define JULIA2D 0
#define CLOUDJULIA 1
#define JULIA4D 2
#define DIRECTIMAGE 4
#define CROSSEYE 8

// half space between eyes in crosseye mode
#define SPACEEYE -0.1
#define CROSSANGLE 0.00001

// 4D rotations angles for Julia 4D
/*__device__ __constant__ float aanglexw;
__device__ __constant__ float aangleyw;
__device__ __constant__ float aanglexy;
__device__ __constant__ float aanglexz;*/

void reshapeFunc(int w, int h);

extern "C" void RunJulia4Drepart(uchar4 *dst, const int imageW, const int imageH,
					const float4 Off,
					const float4 JS,
					const float4 angle,
					const double scale, const double scalei,
					const double xJOff, const double yJOff, const double scaleJ,
					const float xblur, const float yblur, // blur coeff for julia 4D
					const unsigned int maxgropix,
					const unsigned int gropix, const unsigned int nbloc, const unsigned int crn,
					const uchar4 colors, const int frame, const int animationFrame, const int numSMs, const int julia, const int julia4D);

__device__ inline void HSL2RGB(float h, const float sl, const float ll, int *rc, int *gc, int *bc);

__device__ inline int CalcJulia4Dhue(const float xPos, const float yPos, const float zPos, const float wPos, const float4 JS, float *hue);
__device__ inline int CalcMandelBox4Dhue(const float xPos, const float yPos, const float zPos, const float wPos, float *hue);

__device__ inline int CalcMandel4Dcore(const float xPos, const float yPos, const float zPos, const float wPos, const float4 JS, float *hue);

					
extern "C" int inEmulationMode();

__device__ inline void rotate4(float *px, float *py, float *pz, float *pw, const float4 angle);
__device__ inline void rotate3(float *px, float *py, float *pz, const float4 angle);

#endif
