/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
 *
 *   This file is part of Hydra Data Analysis Framework.
 *
 *   Hydra is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   Hydra is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with Hydra.  If not, see <http://www.gnu.org/licenses/>.
 *
 *---------------------------------------------------------------------------*/

/*
 * Ranluxpp.inl
 *
 *  Created on: 28/06/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef RANLUXPP_INL_
#define RANLUXPP_INL_



#include "ranluxpp.h"
#include "mulmod.h"
#include <stdio.h>

namespace hydra {


// modular exponentiation:
// x <- x^n mod (2^576 - 2^240 + 1)
void powmod(uint64_t *x, unsigned long int n){
  uint64_t res[9];
  res[0] = 1;
  for(int i=1;i<9;i++) res[i] = 0;
  while(n){
    if(n&1) mul9x9mod(res, x);
    n >>= 1;
    if(!n) break;
    mul9x9mod(x, x);
  }
  for(int i=0;i<9;i++) x[i] = res[i];
}

const uint64_t *ranluxpp::geta(){
  static const uint64_t
    a[9] = {0x0000000000000001UL, 0x0000000000000000UL, 0x0000000000000000UL,
	    0xffff000001000000UL, 0xffffffffffffffffUL, 0xffffffffffffffffUL,
	    0xffffffffffffffffUL, 0xffffffffffffffffUL, 0xfffffeffffffffffUL};
  return a;
}

ranluxpp::ranluxpp(uint64_t seed, uint64_t p) : _dpos(11), _fpos(24) {
  _x[0] = 1;
  for(int i=1;i<9;i++) _x[i] = 0;
  for(int i=0;i<9;i++) _A[i] = geta()[i];
  powmod(_A, p);
  init(seed);
}

// the core of LCG -- modular mulitplication
void ranluxpp::nextstate(){
  mul9x9mod(_x,_A);
}

void ranluxpp::nextfloats() {
  nextstate(); unpackfloats((float*)_floats); _fpos = 0;
}

void ranluxpp::nextdoubles() {
  nextstate(); unpackdoubles((double*)_doubles); _dpos = 0;
}

// unpack state into single precision format
void ranluxpp::unpackfloats(float *a) {
  const uint32_t m = 0xffffff;
  const float sc = 1.0f/0x1p24f;
  for(int i=0;i<3;i++){
    float *f = a + 8*i;
    uint64_t *t = _x + i*3;
    f[0] = sc * (int32_t)(m & t[0]);
    f[1] = sc * (int32_t)(m & ((t[0]>>24)));
    f[2] = sc * (int32_t)(m & ((t[0]>>48)|(t[1]<<16)));
    f[3] = sc * (int32_t)(m & ((t[1]>>8)));
    f[4] = sc * (int32_t)(m & ((t[1]>>32)));
    f[5] = sc * (int32_t)(m & ((t[1]>>56)|(t[2]<<8)));
    f[6] = sc * (int32_t)(m & ((t[2]>>16)));
    f[7] = sc * (int32_t)(m & ((t[2]>>40)));
  }
}

// unpack state into double precision format
// 52 bits out of possible 53 bits are random
void ranluxpp::unpackdoubles(double *d) {
  const uint64_t
    one = 0x3ff0000000000000, // exponent
    m   = 0x000fffffffffffff; // mantissa
  uint64_t *id = (uint64_t*)d;
  id[ 0] = one | (m & _x[0]);
  id[ 1] = one | (m & ((_x[0]>>52)|(_x[1]<<12)));
  id[ 2] = one | (m & ((_x[1]>>40)|(_x[2]<<24)));
  id[ 3] = one | (m & ((_x[2]>>28)|(_x[3]<<36)));
  id[ 4] = one | (m & ((_x[3]>>16)|(_x[4]<<48)));
  id[ 5] = one | (m & ((_x[4]>> 4)|(_x[5]<<60)));
  id[ 6] = one | (m & ((_x[4]>>56)|(_x[5]<< 8)));
  id[ 7] = one | (m & ((_x[5]>>44)|(_x[6]<<20)));
  id[ 8] = one | (m & ((_x[6]>>32)|(_x[7]<<32)));
  id[ 9] = one | (m & ((_x[7]>>20)|(_x[8]<<44)));
  id[10] = one | (m & _x[8]>>8);

  for(int j=0;j<11;j++) d[j] -= 1;
}



// set the multiplier A to A = a^2048 + 13, a primitive element modulo
// m = 2^576 - 2^240 + 1 to provide the full period m-1 of the sequence.
void ranluxpp::primitive(){
  for(int i=0;i<9;i++) _A[i] = geta()[i];
  powmod(_A, 2048);
  _A[0] += 13;
}

void ranluxpp::init(uint64_t seed){
  uint64_t a[9];
  for(int i=0;i<9;i++) a[i] = _A[i];
  powmod(a, 1UL<<48); powmod(a, 1UL<<48); // skip 2^96 states
  powmod(a, seed); // skip 2^96*seed states
  mul9x9mod(_x, a);
}

// jump ahead by n 24-bit RANLUX numbers
void ranluxpp::jump(uint64_t n){
  uint64_t a[9];
  for(int i=0;i<9;i++) a[i] = geta()[i];
  powmod(a, n);
  mul9x9mod(_x, a);
}


}  // namespace hydra

#endif /* RANLUXPP_INL_ */
