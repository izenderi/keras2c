#include <stdio.h>

unsigned int2hfloat(int x)
{
  unsigned sign = x < 0;
  unsigned absx = ((unsigned)x ^ -sign) + sign; // safe abs(x)
  unsigned tmp = absx, manbits = 0;
  int exp = 0, truncated = 0;

  // calculate the number of bits needed for the mantissa
  while (tmp)
  {
    tmp >>= 1;
    manbits++;
  }

  // half-precision floats have 11 bits in the mantissa.
  // truncate the excess or insert the lacking 0s until there are 11.
  if (manbits)
  {
    exp = 10; // exp bias because 1.0 is at bit position 10
    while (manbits > 11)
    {
      truncated |= absx & 1;
      absx >>= 1;
      manbits--;
      exp++;
    }
    while (manbits < 11)
    {
      absx <<= 1;
      manbits++;
      exp--;
    }
  }

  if (exp + truncated > 15)
  {
    // absx was too big, force it to +/- infinity
    exp = 31; // special infinity value
    absx = 0;
  }
  else if (manbits)
  {
    // normal case, absx > 0
    exp += 15; // bias the exponent
  }

  return (sign << 15) | ((unsigned)exp << 10) | (absx & ((1u<<10)-1));
}

int main(void)
{
  printf(" 0: 0x%04X\n", int2hfloat(0));
  printf("-1: 0x%04X\n", int2hfloat(-1));
  printf("+1: 0x%04X\n", int2hfloat(+1));
  printf("-2: 0x%04X\n", int2hfloat(-2));
  printf("+2: 0x%04X\n", int2hfloat(+2));
  printf("-3: 0x%04X\n", int2hfloat(-3));
  printf("+3: 0x%04X\n", int2hfloat(+3));
  printf("-2047: 0x%04X\n", int2hfloat(-2047));
  printf("+2047: 0x%04X\n", int2hfloat(+2047));
  printf("-2048: 0x%04X\n", int2hfloat(-2048));
  printf("+2048: 0x%04X\n", int2hfloat(+2048));
  printf("-2049: 0x%04X\n", int2hfloat(-2049)); // first inexact integer
  printf("+2049: 0x%04X\n", int2hfloat(+2049));
  printf("-2050: 0x%04X\n", int2hfloat(-2050));
  printf("+2050: 0x%04X\n", int2hfloat(+2050));
  printf("-32752: 0x%04X\n", int2hfloat(-32752));
  printf("+32752: 0x%04X\n", int2hfloat(+32752));
  printf("-32768: 0x%04X\n", int2hfloat(-32768));
  printf("+32768: 0x%04X\n", int2hfloat(+32768));
  printf("-65504: 0x%04X\n", int2hfloat(-65504)); // legal maximum
  printf("+65504: 0x%04X\n", int2hfloat(+65504));
  printf("-65505: 0x%04X\n", int2hfloat(-65505)); // infinity from here on
  printf("+65505: 0x%04X\n", int2hfloat(+65505));
  printf("-65535: 0x%04X\n", int2hfloat(-65535));
  printf("+65535: 0x%04X\n", int2hfloat(+65535));
  return 0;
}