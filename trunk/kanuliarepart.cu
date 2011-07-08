#include <stdio.h>
#include "cutil_inline.h"
#include "kanulia.h"
#include "kanuliacalc.cu"


//  Rotation de quaternion

__device__ inline void rotate4(float *px, float *py, float *pz, float *pw, const float4 angle)
{
	float t;
	if (angle.x != 0. ) {
		t  =   *py * cos(angle.x) + *pz * sin(angle.x);
		*pz = - *py * sin(angle.x) + *pz * cos(angle.x);
		*py = t;
	};
	if (angle.y != 0. ) {
		t   =   *px * cos(angle.y) + *pz * sin(angle.y);
		*pz = - *px * sin(angle.y) + *pz * cos(angle.y);
		*px = t;
	};
	if (angle.z != 0. ) {
		t   =   *pz * cos(angle.z) + *pw * sin(angle.z);
		*pw = - *pz * sin(angle.z) + *pw * cos(angle.z);
		*pz = t;
	};
	if (angle.w != 0. ) {
		t   =   *py * cos(angle.w) + *pw * sin(angle.w);
		*pw = - *py * sin(angle.w) + *pw * cos(angle.w);
		*py = t;
	};
}
__device__ inline void rotate4inv(float *px, float *py, float *pz, float *pw, const float4 angle)
{
	float t;

	if (angle.w != 0. ) {
		t   =   *py * cos(-angle.w) + *pw * sin(-angle.w);
		*pw = - *py * sin(-angle.w) + *pw * cos(-angle.w);
		*py = t;
	};
	if (angle.z != 0. ) {
		t   =   *pz * cos(-angle.z) + *pw * sin(-angle.z);
		*pw = - *pz * sin(-angle.z) + *pw * cos(-angle.z);
		*pz = t;
	};
	if (angle.y != 0. ) {
		t   =   *px * cos(-angle.y) + *pz * sin(-angle.y);
		*pz = - *px * sin(-angle.y) + *pz * cos(-angle.y);
		*px = t;
	};
	if (angle.x != 0. ) {
		t  =   *py * cos(-angle.x) + *pz * sin(-angle.x);
		*pz = - *py * sin(-angle.x) + *pz * cos(-angle.x);
		*py = t;
	};
}

__device__ inline void rotate3(float *px, float *py, float *pz, const float4 angle)
{
	float t;
	if (angle.x != 0. ) {
		t  =    *py * cos(angle.x) + *pz * sin(angle.x);
		*pz = - *py * sin(angle.x) + *pz * cos(angle.x);
		*py =   t;
	};
	if (angle.y != 0. ) {
		t   =   *px * cos(angle.y) + *pz * sin(angle.y);
		*pz = - *px * sin(angle.y) + *pz * cos(angle.y);
		*px =   t;
	};
	if (angle.z != 0. ) {
		t   =   *px * cos(angle.z) - *py * sin(angle.z);
		*py =   *px * sin(angle.z) + *py * cos(angle.z);
		*px =   t;
	};
/*	if (angle.w != 0. ) {
		t   =   *py * cos(angle.w) + *pw * sin(angle.w);
		*pw = - *py * sin(angle.w) + *pw * cos(angle.w);
		*py = t;
	};*/
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

__global__ void Julia4Drepart(uchar4 *dst, const int imageW, const int imageH,
 const float4 Off, const float4 JS, const float4 angle, const float scale, const float scalei,
 const float xJOff, const float yJOff, const float scaleJ,
 const float xblur, const float yblur,
 const unsigned int maxgropix,
 const unsigned int gropix, const unsigned int bloc, const unsigned int crn,
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

//        if (blockIndex >= ((numBlocks/nbloc)+1)*(bloc+1)) break;  // finish
        if (blockIndex >= numBlocks) break;  // finish

        // process this block
        const int ix = blockDim.x * blockX * maxgropix + threadIdx.x * maxgropix + ((bloc * gropix) % maxgropix);
        const int iy = blockDim.y * blockY * maxgropix + threadIdx.y * maxgropix + ((bloc * gropix) / maxgropix) * gropix;

		int r = 0;int g = 0;int b = 0;
		bool seedre = false;bool seedim = false;

		if ((ix < imageW) && (iy < imageH)) {
			int m = 0;
	        if ( (julia<32) && (ix < imageW / julia) && (iy < imageH / julia)) {
			    // Calculate the location
			    const float xPos = (float)ix * scale * julia + Off.x;
				const float yPos = (float)iy * scale * julia + Off.y;

				// Calculate the Mandelbrot index for the current location
				if (abs(JS.x-xPos)+abs(JS.y-yPos) < 2.1 * scale * julia )
				{
					seedre = true; 
				}
				if (!seedre)
				{
					float hue;
//					m = CalcMandelbrot(xPos , yPos);
					m = CalcMandel4Dcore(xPos,  yPos,  JS.z,  JS.w, &hue);
					if (m<=256) HSL2RGB(hue, 0.6, 0.5, &r, &g, &b);
				}
    		} else if (julia4D&& (julia<32) &&((imageW - ix < imageW / julia) && (iy < imageH / julia))) {
			    // Calculate the location
			    const float zPos = (float)(imageW - ix) * scalei * julia + Off.z;
				const float wPos = (float)iy           * scalei * julia  + Off.w;

				// Calculate the Mandelbrot index for the current location
				if (abs(JS.z-zPos)+abs(JS.w-wPos) < 2.1 * scalei * julia )
				{
					seedim = true; 
				}
				if (!seedim)
				{
					float hue;
//					m = CalcMandelbrot(zPos , wPos);
					m = CalcMandel4Dcore(JS.x,  JS.y,  zPos,  wPos, &hue);
					if (m<=256) HSL2RGB(hue, 0.6, 0.5, &r, &g, &b);
				}
			} else {
			    // Calculate the location
			    const float xPos = (float)ix * scaleJ + xJOff;
				const float yPos = (float)iy * scaleJ + yJOff;
/*				const float zPos = (float)0.;
				const float wPos = (float)0.;*/
				// Calculate the Mandelbrot index for the current location
				if (julia4D == JULIA2D)
				{
					m = CalcJulia(xPos, yPos, JS, crn);
				}
				if (julia4D == CLOUDJULIA)
				{
					float dist = 6.0;
					float step = 0.009;

					float ox = (float)ix * scaleJ + xJOff;
					float oy = (float)iy * scaleJ + yJOff;
					float oz = - 3.0;
					float ow = 0.0;
					float dx = sin( 0.7 * step * ( (float)ix + xblur - (imageW/2.)) / ((float) imageW) );
					float dy = sin( 0.7 * step * ( (float)iy + yblur - (imageH/2.)) / ((float) imageW) );
					float dz = step;
					float dw = 0.;
					rotate4(&ox,&oy,&oz,&ow,angle);
					rotate4(&dx,&dy,&dz,&dw,angle);
					int nb = (dist/step);
					m = CloudJulia4D(ox,oy,oz,ow,JS,dx,dy,dz,dw,&r,&g,&b,nb,crn);
				}
				if (julia4D & JULIA4D)
				{
/*					if ((julia4D & CROSSEYE)&&
					   (  (sqrt( (float)((ix-  imageW/4)*(ix-  imageW/4) + (iy-(imageH)/5)*(iy-(imageH)/5) )) < 20.)						// si viseur
						||(sqrt( (float)((ix-3*imageW/4)*(ix-3*imageW/4) + (iy-(imageH)/5)*(iy-(imageH)/5) )) < 20.)))
					{
						r = 255;
						g = 255;
						b = 255;
					}
					else*/
						m = SolidJulia4D(ix-1,iy-1,JS,angle,imageW,imageH,scaleJ,xblur,yblur,&r,&g,&b,xJOff,yJOff,crn,julia4D);
	//				m = SolidMandelBox3D(ix-1,iy-1,JS,angle,imageW,imageH,scaleJ,xblur,yblur,&r,&g,&b,xJOff,yJOff,crn);
				}
    		}
//			m = blockIdx.x;         // uncomment to see scheduling order

            // Convert the Mandelbrot index into a color
            uchar4 color;
//			m = m > 0 ? crn - m : 0;

            if ((julia4D)&&((ix >= imageW / julia) || (iy >= imageH / julia))) {
				color.x = r;
				color.y = g;
				color.z = b;
			} else
			{
				if (seedim||seedre)
				{
					color.x = 150;
					color.y = 250;
					color.z = 250;
				} else {
					color.x = r;
					color.y = g;
					color.z = b;

/*					if (m) {
						m += animationFrame;
						color.x = m * colors.x;
						color.y = m * colors.y;
						color.z = m * colors.z;
					} else {
						color.x = 0;
						color.y = 0;
						color.z = 0;
					}*/
					
				}
			}

			// activer pour voir le calcul progressif
//			if (gropix==1) color.z += 120;
//			if (gropix==2) color.y += 120;
//			if (gropix==4) color.x += 120;
//				
					
			// Output the pixel
			int pixel = imageW * iy + ix;
			if (frame == 0) {
			    color.w = 0;
				if (gropix==1)
					dst[pixel] = color;
				else
					for (int i=0;i<gropix;i++) for (int j=0;j<gropix;j++)
						if ((ix+i<imageW)&&(iy+j<imageH))
							dst[pixel+i+imageW*j] = color;
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


// The host CPU Mandebrot thread spawner
void RunJulia4Drepart(uchar4 *dst, const int imageW, const int imageH,
 const float4 Off,
 const float4 JS,
 const float4 angle,
 const double scale, const double scalei,
 const double xJOff, const double yJOff, const double scaleJ,
 const float xblur, const float yblur,
 const unsigned int maxgropix,
 const unsigned int gropix, const unsigned int bloc, const unsigned int crn,
 const uchar4 colors, const int frame, const int animationFrame, const int numSMs, const int julia, const int julia4D)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW/maxgropix, BLOCKDIM_X), iDivUp(imageH/(maxgropix), BLOCKDIM_Y));

    // zero block counter
//    unsigned int hBlockCounter = (((grid.x)*(grid.y)/nbloc)+1)*(bloc);
    unsigned int hBlockCounter = 0;
    cutilSafeCall( cudaMemcpyToSymbol(blockCounter, &hBlockCounter, sizeof(unsigned int), 0, cudaMemcpyHostToDevice ) );

	int numWorkUnit = numSMs;
	
	Julia4Drepart<<<numWorkUnit, threads>>>(dst, imageW, imageH,
						Off, JS, angle, (float)scale, (float)scalei,
						(float)xJOff, (float)yJOff, (float)scaleJ,
						xblur, yblur,
						maxgropix, gropix, bloc, crn,
						colors, frame, animationFrame, grid.x, (grid.x)*(grid.y), julia, julia4D);

    cutilCheckMsg("Julia4D0_sm13 kernel execution failed.\n");
} // RunJulia4D0


// check if we're running in emulation mode
int inEmulationMode()
{
#ifdef __DEVICE_EMULATION__
    return 1;
#else
    return 0;
#endif
}