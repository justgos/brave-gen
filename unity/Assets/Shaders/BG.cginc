class BGParticle {
	float3 position;
	float val;
};

class BELinePoint {
	float3 position;
	float4 color;
};

/*class BELight {
	float4 pos;
	float4 color;
	float4 atten;
	float4 screenPos;
};*/

float random( float2 p )
{
  float2 r = float2(
    23.1406926327792690,  // e^pi (Gelfond's constant)
     2.6651441426902251); // 2^sqrt(2) (Gelfond–Schneider constant)
  return frac( cos( fmod( 123456789., 1e-7 + 256. * dot(p,r) ) ) );  
}
