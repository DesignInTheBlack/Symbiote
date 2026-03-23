import * as THREE from 'three';
import { getCssVar } from '../../utils/theme';

export function createTechTexture(): THREE.CanvasTexture {
    const size = 512;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    if (ctx) {
        // Background: Transparent
        ctx.fillStyle = 'rgba(0,0,0,0)';
        ctx.clearRect(0, 0, size, size);

        // Tech Grid / Hex Pattern
        // We'll draw a repeating pattern.

        ctx.strokeStyle = getCssVar('--graph-texture-line', 'rgba(255, 255, 255, 0.8)');
        ctx.lineWidth = 2;
        ctx.lineCap = 'round';

        // Simple Hex-like Triangle Mesh which wraps well-ish
        const step = 32;

        ctx.beginPath();
        for (let y = 0; y <= size; y += step) {
            // Horizontal lines (zig-zag?) or just straight grid for "Wireframe" look
            // Let's do a honeycomb-ish offset grid
            const offset = (y / step) % 2 === 0 ? 0 : step / 2;

            for (let x = -step; x <= size; x += step) {
                // Draw Hexagon parts
                // To keep it simple and robust for a sphere texture, a dense triangular grid 
                // often looks very "forcefield" like.

                // Drawing a simple grid for now because full hex math on canvas 
                // without SVG paths can be verbose. 
                // A "Digital Lattice" look.

                // Vertical-ish lines
                ctx.moveTo(x + offset, y);
                ctx.lineTo(x + offset, y + step);

                // Cross lines
                ctx.moveTo(x + offset, y);
                ctx.lineTo(x + offset + step, y + step / 2);

                ctx.moveTo(x + offset + step, y + step / 2);
                ctx.lineTo(x + offset, y + step);
            }
        }
        ctx.stroke();

        // Add some "Data Packets" - filled cells
        ctx.fillStyle = getCssVar('--graph-texture-dot', 'rgba(255, 255, 255, 0.4)');
        for (let i = 0; i < 20; i++) {
            const rx = Math.floor(Math.random() * (size / step)) * step;
            const ry = Math.floor(Math.random() * (size / step)) * step;
            const rOffset = (ry / step) % 2 === 0 ? 0 : step / 2;

            ctx.beginPath();
            ctx.arc(rx + rOffset, ry, step / 3, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    return texture;
}
