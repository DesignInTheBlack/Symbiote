import * as THREE from 'three';

export function createHologramMaterial({ color, map }: { color: string | THREE.Color, map?: THREE.Texture }) {
    const targetColor = new THREE.Color(color);

    // Boost brightness for bloom effect
    const emissiveColor = targetColor.clone().multiplyScalar(3.0);

    const vertexShader = `
        varying vec3 vNormal;
        varying vec3 vViewPosition;
        varying vec2 vUv;

        void main() {
            vUv = uv;
            vNormal = normalize(normalMatrix * normal);
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            vViewPosition = -mvPosition.xyz;
            gl_Position = projectionMatrix * mvPosition;
        }
    `;

    const fragmentShader = `
        uniform vec3 color;
        uniform sampler2D map;
        uniform float useMap;
        uniform float time;
        
        varying vec3 vNormal;
        varying vec3 vViewPosition;
        varying vec2 vUv;

        void main() {
            vec3 normal = normalize(vNormal);
            vec3 viewDir = normalize(vViewPosition);
            
            // Fresnel effect - brighter at edges
            float fresnel = 1.0 - abs(dot(normal, viewDir));
            float rimIntensity = pow(fresnel, 2.0);
            
            // Breathing pulse effect (slow, subtle)
            float pulse = 0.85 + 0.15 * sin(time * 2.0);
            
            // Animated texture UV - creates scanning effect
            vec2 animatedUv = vUv * 3.0;
            animatedUv.y += time * 0.15;
            
            // Sample texture for tech grid pattern
            vec4 texSample = texture2D(map, animatedUv);
            float gridPattern = useMap * texSample.r;
            
            // Scanline effect - horizontal band that moves down
            float scanline = smoothstep(0.0, 0.1, fract(vUv.y * 8.0 - time * 0.5));
            scanline = 1.0 - scanline * 0.3;
            
            // Core visibility (center of sphere)
            float coreAlpha = 0.2 * pulse;
            
            // Combine alpha: rim glow + core + grid pattern
            float alpha = rimIntensity * 0.8 + coreAlpha + gridPattern * 0.4;
            alpha *= scanline;
            alpha = clamp(alpha, 0.0, 1.0);
            
            // Build final color
            vec3 baseGlow = color * (rimIntensity * 3.0 + 0.5) * pulse;
            vec3 gridGlow = color * gridPattern * 2.0 * scanline;
            vec3 finalColor = baseGlow + gridGlow;

            gl_FragColor = vec4(finalColor, alpha);
        }
    `;

    return new THREE.ShaderMaterial({
        uniforms: {
            color: { value: emissiveColor },
            map: { value: map || new THREE.Texture() },
            useMap: { value: map ? 1.0 : 0.0 },
            time: { value: 0.0 }
        },
        vertexShader,
        fragmentShader,
        transparent: true,
        side: THREE.FrontSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
    });
}
