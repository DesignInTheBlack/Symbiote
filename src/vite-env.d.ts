/// <reference types="vite/client" />

// Type declarations for Three.js examples that don't have bundled types
declare module 'three/examples/jsm/postprocessing/UnrealBloomPass' {
    import { Pass } from 'three/examples/jsm/postprocessing/Pass';
    import { Vector2 } from 'three';

    export class UnrealBloomPass extends Pass {
        constructor(resolution: Vector2, strength: number, radius: number, threshold: number);
        strength: number;
        radius: number;
        threshold: number;
        resolution: Vector2;
    }
}

declare module 'three/examples/jsm/postprocessing/Pass' {
    export class Pass {
        enabled: boolean;
        needsSwap: boolean;
        clear: boolean;
        renderToScreen: boolean;
    }
}
