Shader "Custom/ParticleShader" {
	Properties{
		_MainTex("Texture", 2D) = "white" {}
		_Color("Color", Color) = (1,1,1,1)
	}
	SubShader{
		Tags{ "Queue" = "Transparent" "IgnoreProjector" = "True" "RenderType" = "Transparent" }

		Pass{
			//ZTest Always
			Cull Off
			ZWrite Off
			Lighting Off
			Fog{ Mode off }
			//Blend One OneMinusSrcColor
			ColorMask RGB
			//Blend SrcAlpha One
			Blend SrcAlpha OneMinusSrcAlpha
			//BlendOp Add

			CGPROGRAM
			#include "UnityCG.cginc"
			#include "BG.cginc"
			#pragma target 4.0
			#pragma vertex vert
			#pragma fragment frag

			half4 _Color;

			sampler2D _MainTex;
			uniform StructuredBuffer<BGParticle> particles;
			uniform StructuredBuffer<float3> quadPoints;
			uniform StructuredBuffer<float2> quadUVs;
			float4x4 baseTransform;
			// Points with value below this will be discarded
			float cutoffValue;

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				half4 col : COLOR;
				UNITY_FOG_COORDS(1)
				float4 pos : SV_POSITION;
			};

			v2f vert(uint id : SV_VertexID, uint instanceId : SV_InstanceID)
			{
				v2f o;
				BGParticle p = particles[instanceId];
				float3 v = quadPoints[id];
				o.uv = quadUVs[id];
				// Discard near-zero cells and the topmost layer of simulation
				if (p.val < cutoffValue || p.position.y > 150) {
					o.pos = float4(0, 0, 0, 0);
				} else {
					o.pos = mul(UNITY_MATRIX_P,
						mul(UNITY_MATRIX_V,
							mul(baseTransform,
								float4(p.position, 1)
							)) + float4(quadPoints[id] * float3(10.0, 10.0, 1), 0));
				}
				o.col = float4(1, 1, 1, min(p.val*5.0, 1));
				UNITY_TRANSFER_FOG(o, o.pos);
				return o;
			}

			fixed4 frag(v2f i) : SV_Target
			{
				fixed4 pc = tex2D(_MainTex, i.uv);
				fixed4 d = _Color * pc * i.col;
				return d;
			}

			ENDCG
		}
	}
}
