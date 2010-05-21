#include <stdio.h>
#include "cutil_inline.h"
#include "kanulia.h"

// The dimensions of the thread block
#define BLOCKDIM_X 32
#define BLOCKDIM_Y 16

#define ABS(n) ((n) < 0 ? -(n) : (n))


// return the argument of a complex number
template<class T>
__device__ inline float arg( T re, T im )
{
	float pi = 3.14159;
	float a = 0.;
	if ((re>0.)&&(im>0.)) a= atan(im/re);
	if ((re>0.)&&(im<0.)) a=2.*pi-atan(-im/re);
	if ((re<0.)&&(im>0.)) a=pi-atan(-im/re);
	if ((re<0.)&&(im<0.)) a=pi+atan(im/re);
	if ((re>0.)&&(im==0.)) a= 0.;
	if ((re==0.)&&(im>0.)) a= pi / 2.;
	if ((re<0.)&&(im==0.)) a= pi;
	if ((re==0.)&&(im<0.)) a= ( 3. * pi ) / 2.;
	return a/(2.*pi);
}

// Given H,S,L in range of 0-1
// Returns a Color (RGB struct) in range of 0-255
__device__ inline void HSL2RGB(float h, const float sl, const float ll, int *rc, int *gc, int *bc)
{
    float v,r,g,b,l = ll;
	if ( ll < 0. ) l = 0;
	if ( ll > 1. ) l = 1;
    r = l;   // default to gray
    g = l;
    b = l;
    v = (l <= 0.5) ? (l * (1.0 + sl)) : (l + sl - l * sl);
    if (v > 0)
    {
          float m;
          float sv;
          int sextant;
          float fract, vsf, mid1, mid2;

          m = l + l - v;
          sv = (v - m ) / v;
          h *= 6.0;
          sextant = (int)h;
          fract = h - sextant;
          vsf = v * sv * fract;
          mid1 = m + vsf;
          mid2 = v - vsf;
          switch (sextant)
          {
                case 0:
                      r = v;
                      g = mid1;
                      b = m;
                      break;
                case 1:
                      r = mid2;
                      g = v;
                      b = m;
                      break;
                case 2:
                      r = m;
                      g = v;
                      b = mid1;
                      break;
                case 3:
                      r = m;
                      g = mid2;
                      b = v;
                      break;
                case 4:
                      r = mid1;
                      g = m;
                      b = v;
                      break;
                case 5:
                      r = v;
                      g = m;
                      b = mid2;
                      break;
          }
    }
    *rc = r * 255.0;
    *gc = g * 255.0;
    *bc = b * 255.0;
}

// The core Mandelbrot CUDA GPU calculation function
// Unrolled version
template<class T>
__device__ inline int CalcMandelbrot(const T xPos, const T yPos)
{
    T y = yPos;
    T x = xPos;
    T yy = y * y;
    T xx = x * x;
    int i = crn;

    do {
		// Iteration 1
		if (xx + yy > T(4.0))
			return i - 1;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 2
		if (xx + yy > T(4.0))
			return i - 2;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 3
		if (xx + yy > T(4.0))
			return i - 3;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 4
		if (xx + yy > T(4.0))
			return i - 4;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 5
		if (xx + yy > T(4.0))
			return i - 5;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 6
		if (xx + yy > T(4.0))
			return i - 6;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 7
		if (xx + yy > T(4.0))
			return i - 7;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 8
		if (xx + yy > T(4.0))
			return i - 8;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 9
		if (xx + yy > T(4.0))
			return i - 9;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 10
		if (xx + yy > T(4.0))
			return i - 10;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 11
		if (xx + yy > T(4.0))
			return i - 11;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 12
		if (xx + yy > T(4.0))
			return i - 12;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 13
		if (xx + yy > T(4.0))
			return i - 13;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 14
		if (xx + yy > T(4.0))
			return i - 14;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 15
		if (xx + yy > T(4.0))
			return i - 15;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 16
		if (xx + yy > T(4.0))
			return i - 16;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 17
		if (xx + yy > T(4.0))
			return i - 17;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 18
		if (xx + yy > T(4.0))
			return i - 18;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 19
		if (xx + yy > T(4.0))
			return i - 19;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;

		// Iteration 20
        i -= 20;
		if ((i <= 0) || (xx + yy > T(4.0)))
			return i;
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;
    } while (1);
} // CalcMandelbrot

// The core Julia CUDA GPU calculation function

// Unrolled version
template<class T>
__device__ inline int CalcJulia(const T xPos, const T yPos)
{
    T y = yPos;
    T x = xPos;
    T yy = y * y;
    T xx = x * x;
    int i = crn;

    do {
		// Iteration 1
		if (xx + yy > T(4.0))
			return i - 1;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 2
		if (xx + yy > T(4.0))
			return i - 2;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 3
		if (xx + yy > T(4.0))
			return i - 3;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 4
		if (xx + yy > T(4.0))
			return i - 4;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 5
		if (xx + yy > T(4.0))
			return i - 5;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 6
		if (xx + yy > T(4.0))
			return i - 6;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 7
		if (xx + yy > T(4.0))
			return i - 7;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 8
		if (xx + yy > T(4.0))
			return i - 8;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 9
		if (xx + yy > T(4.0))
			return i - 9;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 10
		if (xx + yy > T(4.0))
			return i - 10;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 11
		if (xx + yy > T(4.0))
			return i - 11;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 12
		if (xx + yy > T(4.0))
			return i - 12;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 13
		if (xx + yy > T(4.0))
			return i - 13;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 14
		if (xx + yy > T(4.0))
			return i - 14;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 15
		if (xx + yy > T(4.0))
			return i - 15;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 16
		if (xx + yy > T(4.0))
			return i - 16;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 17
		if (xx + yy > T(4.0))
			return i - 17;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 18
		if (xx + yy > T(4.0))
			return i - 18;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 19
		if (xx + yy > T(4.0))
			return i - 19;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;

		// Iteration 20
        i -= 20;
		if ((i <= 0) || (xx + yy > T(4.0)))
			return i;
        y = x * y * T(2.0) + yJS;
        x = xx - yy + xJS;
        yy = y * y;
        xx = x * x;
    } while (1);
} // CalcJulia
// The core Julia CUDA GPU calculation function

template<class T>
__device__ inline int CalcJulia4D(const T xPos, const T yPos, const T zPos, const T wPos)
{
    T x = xPos;T y = yPos;T z = zPos;T w = wPos;
    T xx = x * x;
    T yy = y * y;
    T zz = z * z;
    T ww = w * w;
    int i = crn;

//	if (y>0) return i;
    do {
		i--;
		if (xx + yy + zz + ww > T(4.0))
			return i;
        z = x * z * T(2.0) + zJS;
        w = x * w * T(2.0) + wJS;
        y = x * y * T(2.0) + yJS;
        x = xx - yy - zz - ww + xJS;
        xx = x * x;
        yy = y * y;
        zz = z * z;
        ww = w * w;
    } while (i);
    return 0;
} // CalcJulia4D

template<class T>
__device__ inline int CalcJulia4Dhue(const T xPos, const T yPos, const T zPos, const T wPos, float *hue)
{
    T x = xPos;T y = yPos;T z = zPos;T w = wPos;
    T xx = x * x;
    T yy = y * y;
    T zz = z * z;
    T ww = w * w;

    int i = crn;
	int huenb = 7;

	if (huenb>i) huenb = i;

    do {
		i--;
		huenb--;
		if (huenb==0) *hue = arg(y,z);

		if (xx + yy + zz + ww > T(4.0))
		{
//			*hue = 0.5 + cos((x+y+z+w)/4.)/2.;
			return i;
		}
        z = x * z * T(2.0) + zJS;
        w = x * w * T(2.0) + wJS;
        y = x * y * T(2.0) + yJS;
        x = xx - yy - zz - ww + xJS;
        xx = x * x;
        yy = y * y;
        zz = z * z;
        ww = w * w;
    } while (i);
    return 0;
} // CalcJulia4Dhue

template<class T>
__device__ inline int CalcJulia4Dcore(const T xPos, const T yPos, const T zPos, const T wPos, float *hue)
{
    T x = xPos;T y = yPos;T z = zPos;T w = wPos;
    T xx = x * x;
    T yy = y * y;
    T zz = z * z;
    T ww = w * w;

    int i = 0;

    do {
		i++;

		if (xx + yy + zz + ww > T(4.0))
		{
			*hue =(float)(i)/256.0;
			while (*hue>1.0) *hue -= 1.0;
			return i;
		}
        z = x * z * T(2.0) + zJS;
        w = x * w * T(2.0) + wJS;
        y = x * y * T(2.0) + yJS;
        x = xx - yy - zz - ww + xJS;
        xx = x * x;
        yy = y * y;
        zz = z * z;
        ww = w * w;
    } while (i<=256);
	*hue = 0.3;
    return i;
} // CalcJulia4Dhue


// The core Julia CUDA GPU calculation function

template<class T>
__device__ int CloudJulia4D(const T ox, const T oy, const T oz, const T ow, const T dx, const T dy, const T dz, const T dw, int *r, int *g, int *b, int nb)
{
	T ret = 0;
	T x=ox;
	T y=oy;
	T z=oz;
	T w=ow;
	int c=nb;
	do {
		x += dx;
		y += dy;
		z += dz;
		w += dw;

		if (CalcJulia4D(x, y, z, w)==0) ret +=1;
	} while (c--);

	if (ret>255) ret=255;
	if (ret==0) {
		*r = 0;
		*g = 0;
		*b = 0;
	} else {
		*r = ret;
		*g = ret;
		*b = 155;
	}
	return ret;
} // CalcJulia

// The core Julia CUDA GPU calculation function
template<class T>
__device__ int SolidJulia4D(const int ix, const int iy, const int d_imageW, const int d_imageH, const T scaleJ,
	const float xblur, const float yblur, int *r, int *g, int *b, const T xJOff, const T yJOff)
{
	//hue color
	float hue;
	T dist = 6.0;
	T step = 0.007;

	T x = (T)ix * scaleJ + xJOff;
	T y = (T)iy * scaleJ + yJOff;
	T z = - 3.0;
	T w = 0.0;
	T dx = sin( 0.7 * step * ( (float) ix + xblur - (d_imageW/2.)) / ((float) d_imageW) );
	T dy = sin( 0.7 * step * ( (float) iy + yblur - (d_imageH/2.)) / ((float) d_imageW) );
	T dz = step;
	T dw = 0.;
	rotate4(&x,&y,&z,&w);
	rotate4(&dx,&dy,&dz,&dw);
	int nb = (dist/step);

	T x0 = 0.0;T y0 = 1.0;T z0 = 0.0;T w0 = 0.0;// normal is the secant plan's normal
	T x1 = step;T y1 = 0.0;T z1 = 0.0;T w1 = 0.0;
	T x2 = 0.0;T y2 = step;T z2 = 0.0;T w2 = 0.0;
	T x3 = 0.0;T y3 = 0.0;T z3 = 0.0;T w3 = 1.0;

	rotate4(&x1,&y1,&z1,&w1);
	rotate4(&x2,&y2,&z2,&w2);
	rotate4(&x3,&y3,&z3,&w3);

	T xl = 1.;
	T yl = -1.;
	T zl = 1.;
	T wl = 0.;
	rotate4(&xl,&yl,&zl,&wl);

	T ddx=dx;
	T ddy=dy;
	T ddz=dz;
	T ddw=dw;
	int c=nb;
	bool out = true; // if ray is out main c=0
	bool hit = false; // if the ray hit the "inside"
	do {
		// if inside empty aera
		if ( y < 0.)
		{
			// then if going away
			if (dy < 0.)
			{
				// go away
				x = 4.0;y = 0.;z = 0.;w = 0.;
				c = 0;
			}
			else
			{
				// hit the surface
				T dhit = -y/dy;
				x += dx * dhit;
				y += dy * dhit;
				z += dz * dhit;
				w += dw * dhit;
				if (CalcJulia4Dcore(x,  y,  z,  w, &hue)>=crn)
				{
					c=0; // stop, we hit the inside
					hit = true;
					out = false;
				}
			}
		}
		else
		{
			x += dx;y += dy;z += dz;w += dw;

			if (CalcJulia4D(x, y, z, w)==0)
			{
				// ray is not out. we ll see if normal is out now
				out=false;
				c=12;

				// for normal 3D
				x1=x + x1;
				y1=y + y1;
				z1=z + z1;
				w1=w + w1;

				x2=x + x2;
				y2=y + y2;
				z2=z + z2;
				w2=w + w2;

				ddx=dx;ddy= dy;ddz=dz;ddw=dw;
				T d1x=dx;T d1y=dy;T d1z=dz;T d1w=dw;
				T d2x=dx;T d2y=dy;T d2z=dz;T d2w=dw;
				int in=0,in1=0,in2=0;//,in3=0;

				// place les 2 rayons pour les normales contre la forme
				if (CalcJulia4D(x1, y1, z1, w1)==0)
				{
					do {
						x1 -= d1x;y1 -= d1y;z1 -= d1z;w1 -= d1w;
						if (x1*x1 + y1*y1 + z1*z1 + w1*w1 > T(4.0)) out=true;
					} while ((CalcJulia4D(x1, y1, z1, w1) == 0) && (!out) );
				} else {
					do {
						x1 += d1x;y1 += d1y;z1 += d1z;w1 += d1w;
						if (x1*x1 + y1*y1 + z1*z1 + w1*w1 > T(4.0)) out=true;
					} while ((CalcJulia4D(x1, y1, z1, w1) != 0) && (!out) );
				}
				if (CalcJulia4D(x2, y2, z2, w2)==0)
				{
					do {
						x2 -= d2x;y2 -= d2y;z2 -= d2z;w2 -= d2w;
						if (x2*x2 + y2*y2 + z2*z2 + w2*w2 > T(4.0)) out=true;
					} while ((CalcJulia4D(x2, y2, z2, w2) == 0) && (!out) );
				} else {
					do {
						x2 += d2x;y2 += d2y;z2 += d2z;w2 += d2w;
						if (x2*x2 + y2*y2 + z2*z2 + w2*w2 > T(4.0)) out=true;
					} while ((CalcJulia4D(x2, y2, z2, w2) != 0) && (!out) );
				}

				if (!out) {
					do {
						in  = CalcJulia4Dhue(x,  y,  z,  w, &hue);
						in1 = CalcJulia4D(x1, y1, z1, w1);
						in2 = CalcJulia4D(x2, y2, z2, w2);
						if (in==0) {
							x -= ddx;y -= ddy;z -= ddz;w -= ddw;
						} else {
							x += ddx;y += ddy;z += ddz;w += ddw;
						}
						if (in1==0) {
							x1 -= d1x;y1 -= d1y;z1 -= d1z;w1 -= d1w;
						} else {
							x1 += d1x;y1 += d1y;z1 += d1z;w1 += d1w;
						}
						if (in2==0) {
							x2 -= d2x;y2 -= d2y;z2 -= d2z;w2 -= d2w;
						} else {
							x2 += d2x;y2 += d2y;z2 += d2z;w2 += d2w;
						}
						ddx /= 2.0;ddy /= 2.0;ddz /= 2.0;ddw /= 2.0;
						d1x /= 2.0;d1y /= 2.0;d1z /= 2.0;d1w /= 2.0;
						d2x /= 2.0;d2y /= 2.0;d2z /= 2.0;d2w /= 2.0;
					} while (c-->0);
				} else c=1;
			}
		}
	} while (c-->0);

	if (out) {
		*r = 1;
		*g = 1;
		*b = 1;
	} else {
		if (!hit)
		{
			// computing vector
			x1 -= x;y1 -= y;z1 -= z;w1 -= w;
			x2 -= x;y2 -= y;z2 -= z;w2 -= w;
			// vector product for normal
//	3D Normal in space vue
//			x0 = x1 * x2 - y1 * y2 - z1 * z2 - w1* w2;
//			y0 = x1 * y2 + y1 * x2 + z1 * w2 - w1* z2;
//			z0 = x1 * z2 + z1 * x2 + w1 * y2 - y1* w2;
//			w0 = x1 * w2 + w1 * x2 + y1 * z2 - z1* y2;
//	4D Normal
			x0 = y1*(w2*z3-z2*w3)+y2*(z1*w3-w1*z3)+y3*(w1*z2-z1*w2);
			y0 = x1*(z2*w3-w2*z3)+x2*(w1*z3-z1*w3)+x3*(z1*w2-w1*z2);
			z0 = x1*(w2*y3-y2*w3)+x2*(y1*w3-w1*y3)+x3*(w1*y2-y1*w2);
			w0 = x1*(y2*z3-z2*y3)+x2*(z1*y3-y1*z3)+x3*(y1*z2-z1*y2);
//	3D Normal in space xyz
//			x0 = y1 * z2 - z1 * y2;
//			y0 = z1 * x2 - x1 * z2;
//			z0 = x1 * y2 - y1 * x2;
//			w0 = 0.;
		}

		// Normalisation
		T nd=sqrt(dx*dx+dy*dy+dz*dz+dw*dw);
		T n0=sqrt(x0*x0+y0*y0+z0*z0+w0*w0);
		T nl=sqrt(xl*xl+yl*yl+zl*zl+wl*wl);
		dx/=nd;dy/=nd;dz/=nd;dw/=nd;
		x0/=n0;y0/=n0;z0/=n0;w0/=n0;
		xl/=nl;yl/=nl;zl/=nl;wl/=nl;

		// angle of direction / normal
		T anv = (x0 * dx + y0 *dy + z0 *dz + w0 *dw);
		if (anv<0.) anv=0.;
		// angle of light / normal

		// angle of light direction / normal
		T anl = -(x0* xl + y0* yl + z0*zl + w0*wl);
		if (anl<0.) anl=0.;

		// radiance
		T anr = 0.;
		if ( xl*x0 + yl*y0 + zl*z0 + wl*w0 < 0. )
		{
			T xr=xl+2.*x0;T yr=yl+2.*y0;T zr=zl+2.*z0;T wr=wl+2.*w0;
			T nr=sqrt(xr*xr+yr*yr+zr*zr+wr*wr);
			xr/=nr;yr/=nr;zr/=nr;wr/=nr;
			anr = -0.85 -(xr*dx + yr*dy + zr*dz + wr*dw);
		}
		if ( anr < 0. ) anr=0.;
		anr *= 9.;
		if ( anr > 1. ) anr=1.;
		T li = anl*0.7+0.1;
		if (!hit)
			HSL2RGB(hue, 0.6, li + (1. - li)*anr*anr, r, g, b);
		else
			HSL2RGB(hue, 0.6, 0.5, r, g, b);
		//+(anr*anr*anr)/3.
	}
	return out;
} // CalcJulia


// Determine if two pixel colors are within tolerance
__device__ inline int CheckColors(const uchar4 &color0, const uchar4 &color1)
{
	int x = color1.x - color0.x;
	int y = color1.y - color0.y;
	int z = color1.z - color0.z;
	return (ABS(x) > 10) || (ABS(y) > 10) || (ABS(z) > 10);
} // CheckColors


// Increase the grid size by 1 if the image width or height does not divide evenly
// by the thread block dimensions
inline int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
} // iDivUp
