#include <stdio.h>
#include "cutil_inline.h"
#include "kanulia.h"
#include "kanuliacalc.cu"


//  Rotation de quaternion
template<class T>
__device__ inline void rotate4(T *px, T *py, T *pz, T *pw)
{
	T t;
	if (aanglexw != 0. ) {
		t  =   *py * cos(aanglexw) + *pz * sin(aanglexw);
		*pz = - *py * sin(aanglexw) + *pz * cos(aanglexw);
		*py = t;
	};
	if (aangleyw != 0. ) {
		t   =   *px * cos(aangleyw) + *pz * sin(aangleyw);
		*pz = - *px * sin(aangleyw) + *pz * cos(aangleyw);
		*px = t;
	};
	if (aanglexy != 0. ) {
		t   =   *pz * cos(aanglexy) + *pw * sin(aanglexy);
		*pw = - *pz * sin(aanglexy) + *pw * cos(aanglexy);
		*pz = t;
	};
	if (aanglexz != 0. ) {
		t   =   *py * cos(aanglexz) + *pw * sin(aanglexz);
		*pw = - *py * sin(aanglexz) + *pw * cos(aanglexz);
		*py = t;
	};
}

// The Julia4D CUDA GPU thread function

/*
    Version using software scheduling of thread blocks.

    The idea here is to launch of fixed number of worker blocks to fill the
    machine, and have each block loop over the available work until it is all done.

    We use a counter in global memory to keep track of which blocks have been
    completed. The counter is incremented atomically by each worker block.

    This method can achieve higher performance when blocks take a wide range of
    different times to complete.
*/

__device__ unsigned int blockCounter;   // global counter, initialized to zero before kernel launch

template<class T>
__global__ void Julia4Drepart(uchar4 *dst, const int imageW, const int imageH,
 const T xOff, const T yOff, const T scale,
 const T xJOff, const T yJOff, const T scaleJ,
 const float xblur, const float yblur,
 const uchar4 colors, const int frame,
 const int animationFrame, const int gridWidth, const int numBlocks, const int julia, const int julia4D)
{
    __shared__ unsigned int blockIndex;
    __shared__ unsigned int blockX, blockY;

    // loop until all blocks completed
    while(1) {
        if ((threadIdx.x==0) && (threadIdx.y==0)) {
            // get block to process
            blockIndex = atomicAdd(&blockCounter, 1);
            blockX = blockIndex % gridWidth;            // note: this is slow, but only called once per block here
            blockY = blockIndex / gridWidth;
        }
#ifndef __DEVICE_EMULATION__        // device emu doesn't like syncthreads inside while()
        __syncthreads();
#endif

        if (blockIndex >= numBlocks) break;  // finish

        // process this block
        const int ix = blockDim.x * blockX + threadIdx.x;
        const int iy = blockDim.y * blockY + threadIdx.y;

		int r = 0;int g = 0;int b = 0;
		bool seedre = false;bool seedim = false;

		if ((ix < imageW) && (iy < imageH)) {
			int m = 0;
	        if ((ix < imageW / julia) && (iy < imageH / julia)) {
			    // Calculate the location
			    const T xPos = (T)ix * scale * julia + xOff;
				const T yPos = (T)iy * scale * julia + yOff;

				// Calculate the Mandelbrot index for the current location
				if (abs(xJS-xPos)+abs(yJS-yPos) < 2.1 * scale * julia )
				{
					seedre = true; 
				}
				if (abs(zJS-xPos)+abs(wJS-yPos) < 2.1 * scale * julia )
				{
					seedim = true; 
				}
				if (!(seedre||seedim))
				{
					m = CalcMandelbrot<T>(xPos, yPos);
				}
    		} else {
			    // Calculate the location
			    const T xPos = (T)ix * scaleJ + xJOff;
				const T yPos = (T)iy * scaleJ + yJOff;
/*				const T zPos = (T)0.;
				const T wPos = (T)0.;*/
				// Calculate the Mandelbrot index for the current location
				if (julia4D == 0)
				{
					m = CalcJulia<T>(xPos, yPos);
				}
				if (julia4D == 1)
				{
					T dist = 6.0;
					T step = 0.009;

					T ox = (T)ix * scaleJ + xJOff;
					T oy = (T)iy * scaleJ + yJOff;
					T oz = - 3.0;
					T ow = 0.0;
					T dx = sin( 0.7 * step * ( (T)ix + xblur - (imageW/2.)) / ((float) imageW) );
					T dy = sin( 0.7 * step * ( (T)iy + yblur - (imageH/2.)) / ((float) imageW) );
					T dz = step;
					T dw = 0.;
					rotate4(&ox,&oy,&oz,&ow);
					rotate4(&dx,&dy,&dz,&dw);
					int nb = (dist/step);
					m = CloudJulia4D<T>(ox,oy,oz,ow,dx,dy,dz,dw,&r,&g,&b,nb);
				}
				if (julia4D == 2)
				{
					m = SolidJulia4D<T>(ix-1,iy-1,imageW,imageH,scaleJ,xblur,yblur,&r,&g,&b,xJOff,yJOff);
				}
    		}
//			m = blockIdx.x;         // uncomment to see scheduling order

            // Convert the Mandelbrot index into a color
            uchar4 color;
			m = m > 0 ? crn - m : 0;

            if ((julia4D)&&((ix >= imageW / julia) || (iy >= imageH / julia))) {
				color.x = r;
				color.y = g;
				color.z = b;
			} else
			{
				if (seedim)
				{
					color.x = 250;
					color.y = 150;
					color.z = 150;
				}
				if (seedre)
				{
					color.x = 150;
					color.y = 250;
					color.z = 250;
				}
				
				if (!(seedre||seedim))
				{
					if (m) {
						m += animationFrame;
						color.x = m * colors.x;
						color.y = m * colors.y;
						color.z = m * colors.z;
					} else {
						color.x = 0;
						color.y = 0;
						color.z = 0;
					}
				}
			}

			// Output the pixel
			int pixel = imageW * iy + ix;
			if (frame == 0) {
			    color.w = 0;
			    dst[pixel] = color;
			} else {
			    int frame1 = frame + 1;
			    int frame2 = frame1 / 2;
			    dst[pixel].x = (dst[pixel].x * frame + color.x + frame2) / frame1;
			    dst[pixel].y = (dst[pixel].y * frame + color.y + frame2) / frame1;
			    dst[pixel].z = (dst[pixel].z * frame + color.z + frame2) / frame1;
			}
        }

    }

} // Julia4D0


// The Julia4D secondary AA pass CUDA GPU thread function
/*template<class T>
__global__ void Julia4D1_sm13(uchar4 *dst, const int imageW, const int imageH, const T xOff, const T yOff,
  const T scale, const T xJOff, const T yJOff, const T scaleJ, const uchar4 colors,
  const int frame, const int animationFrame, const int gridWidth, const int numBlocks, const int julia, const int julia4D)
{
    __shared__ unsigned int blockIndex;
    __shared__ unsigned int blockX, blockY;

    // loop until all blocks completed
    while(1) {
        if ((threadIdx.x==0) && (threadIdx.y==0)) {
            // get block to process
            blockIndex = atomicAdd(&blockCounter, 1);
            blockX = blockIndex % gridWidth;            // note: this is slow, but only called once per block here
            blockY = blockIndex / gridWidth;
        }
#ifndef __DEVICE_EMULATION__
        __syncthreads();
#endif

        if (blockIndex >= numBlocks) break;  // finish

        // process this block
        const int ix = blockDim.x * blockX + threadIdx.x;
        const int iy = blockDim.y * blockY + threadIdx.y;
		int r = 0;int g = 0;int b = 0;

        if ((ix < imageW) && (iy < imageH)) {
		    // Get the current pixel color
 		    int pixel = imageW * iy + ix;
		    uchar4 pixelColor = dst[pixel];
		    int count = 0;

		    // Search for pixels out of tolerance surrounding the current pixel
		    if (ix > 0)
			    count += CheckColors(pixelColor, dst[pixel - 1]);
		    if (ix + 1 < imageW)
			    count += CheckColors(pixelColor, dst[pixel + 1]);
		    if (iy > 0)
			    count += CheckColors(pixelColor, dst[pixel - imageW]);
		    if (iy + 1 < imageH)
			    count += CheckColors(pixelColor, dst[pixel + imageW]);
		    if (count) {

				int m = 0;
				if ((ix < imageW / julia) && (iy < imageH / julia)) {
				    // Calculate the location
				    const T xPos = (T)ix * scale * julia + xOff;
					const T yPos = (T)iy * scale * julia + yOff;

					// Calculate the Mandelbrot index for the current location
					m = CalcMandelbrot<T>(xPos, yPos);
    			} else {
				    // Calculate the location
				    const T xPos = (T)ix * scaleJ + xJOff;
					const T yPos = (T)iy * scaleJ + yJOff;
//					const T zPos = (T)0.;
//					const T wPos = (T)0.;


					// Calculate the Mandelbrot index for the current location
					switch (julia4D) {
						case 0:
							m = CalcJulia<T>(xPos, yPos);
							break;
						case 1:
						case 2:
							T dist = 6.0;
							T step = 0.009;

							T ox = (T)ix * scaleJ + xJOff;
							T oy = (T)iy * scaleJ + yJOff;
							T oz = - 3.0;
							T ow = 0.0;
							T dx = sin( 0.7 * step * ( ix - (imageW/2.)) / ((float) imageW) );
							T dy = sin( 0.7 * step * ( iy - (imageH/2.)) / ((float) imageW) );
							T dz = step;
							T dw = 0.;
							rotate4(&ox,&oy,&oz,&ow);
							rotate4(&dx,&dy,&dz,&dw);
							int nb = (dist/step);
							if (julia4D==1)
								m = CloudJulia4D<T>(ox,oy,oz,ow,dx,dy,dz,dw,&r,&g,&b,nb);
							if (julia4D==2)
							{
								m = SolidJulia4D<T>(ix,iy,imageW,imageH,scaleJ,&r,&g,&b,xJOff,yJOff);
							}
							break;
					}
    			}
//				m = blockIdx.x;         // uncomment to see scheduling order
				m = m > 0 ? crn - m : 0;

				// Convert the Mandelbrot index into a color
				uchar4 color;

				if ((julia4D)&&((ix >= imageW / julia) || (iy >= imageH / julia))) {
					color.x = r;
					color.y = g;
					color.z = b;
				} else
				{
					if (m) {
						m += animationFrame;
						color.x = m * colors.x;
						color.y = m * colors.y;
						color.z = m * colors.z;
					} else {
						color.x = 0;
						color.y = 0;
						color.z = 0;
					}
				}

				// Output the pixel
				int frame1 = frame + 1;
				int frame2 = frame1 / 2;
				dst[pixel].x = (pixelColor.x * frame + color.x + frame2) / frame1;
				dst[pixel].y = (pixelColor.y * frame + color.y + frame2) / frame1;
				dst[pixel].z = (pixelColor.z * frame + color.z + frame2) / frame1;
		    }
        }
    }

}*/ // Julia4D1

// The host CPU Mandebrot thread spawner
void RunJulia4Drepart(uchar4 *dst, const int imageW, const int imageH,
 const double xOff, const double yOff, const double scale,
 const double xJOff, const double yJOff, const double scaleJ,
 const float xblur, const float yblur,
 const uchar4 colors, const int frame, const int animationFrame, const int mode, const int numSMs, const int julia, const int julia4D)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    // zero block counter
    unsigned int hBlockCounter = 0;
    cutilSafeCall( cudaMemcpyToSymbol(blockCounter, &hBlockCounter, sizeof(unsigned int), 0, cudaMemcpyHostToDevice ) );

	int numWorkUnit = numSMs;
	
	switch(mode) {
	default:
	case 0:
	    Julia4Drepart<float><<<numWorkUnit, threads>>>(dst, imageW, imageH,
						(float)xOff, (float)yOff, (float)scale,
						(float)xJOff, (float)yJOff, (float)scaleJ,
						xblur, yblur,
						colors, frame, animationFrame, grid.x, grid.x*grid.y, julia, julia4D);
	    break;
	case 1:
		Julia4Drepart<double><<<numWorkUnit, threads>>>(dst, imageW, imageH,
						xOff, yOff, scale,
						xJOff, yJOff, scaleJ,
						xblur, yblur,
						colors, frame, animationFrame, grid.x, grid.x*grid.y, julia, julia4D);
		break;
	}
    cutilCheckMsg("Julia4D0_sm13 kernel execution failed.\n");
} // RunJulia4D0

// The host CPU Mandebrot thread spawner
/*void RunJulia4D1_sm13(uchar4 *dst, const int imageW, const int imageH, const double xOff, const double yOff, const double scale, const double xJOff, const double yJOff, const double scaleJ, const uchar4 colors, const int frame, const int animationFrame, const int mode, const int numSMs, const int julia, const int julia4D)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    // zero block counter
    unsigned int hBlockCounter = 0;
    cutilSafeCall( cudaMemcpyToSymbol(blockCounter, &hBlockCounter, sizeof(unsigned int), 0, cudaMemcpyHostToDevice ) );

	int numWorkUnit = numSMs;
	
	switch(mode) {
	default:
    case 0:
		Julia4D1_sm13<float><<<numWorkUnit, threads>>>(dst, imageW, imageH, (float)xOff, (float)yOff, (float)scale, (float)xJOff, (float)yJOff, (float)scaleJ, colors, frame, animationFrame, grid.x, grid.x*grid.y, julia, julia4D);
        break;
    case 1:
		Julia4D1_sm13<double><<<numWorkUnit, threads>>>(dst, imageW, imageH, xOff, yOff, scale, xJOff, yJOff, scaleJ, colors, frame, animationFrame, grid.x, grid.x*grid.y, julia, julia4D);
        break;
    }

    cutilCheckMsg("Julia4D1_sm13 kernel execution failed.\n");
}*/ // RunMandelbrot1


// check if we're running in emulation mode
int inEmulationMode()
{
#ifdef __DEVICE_EMULATION__
    return 1;
#else
    return 0;
#endif
}