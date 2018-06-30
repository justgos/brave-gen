Shader "Hidden/FilmGrain"
{
	Properties
	{
		_MainTex ("Render Input", 2D) = "white" {}
	}
	SubShader
	{
		// No culling or depth
		Cull Off ZWrite Off ZTest Always

		Pass
		{
			CGPROGRAM
			#pragma vertex vert_img
			#pragma fragment frag
			
			#include "UnityCG.cginc"

			float rand(float3 co)
			{
				return frac(sin(dot(co.xyz, float3(12.9898, 78.233, 45.5432))) * 43758.5453);
			}

			sampler2D _MainTex;

			fixed4 frag (v2f_img i) : COLOR
			{
				fixed4 col = tex2D(_MainTex, i.uv);
				col.rgb = clamp(col.rgb+rand(float3(i.uv.x, i.uv.y, 0)), 0, 1);
				return col;
			}
			ENDCG
		}
	}
}
