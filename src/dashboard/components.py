import streamlit as st
import streamlit.components.v1 as components

def render_3d_spectrogram_hud(prediction: str, spoof_prob: float, bonafide_prob: float, latency_ms: float):
    color = "#e74c3c" if prediction == "spoof" else "#2ecc71"
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <style>
            body {{ margin: 0; overflow: hidden; background: #0b0f19; font-family: 'Segoe UI', monospace; }}
            div#hud-container {{ position: relative; width: 100%; height: 350px; border-radius: 8px; overflow: hidden; border: 1px solid #1e293b; }}
            div#overlay {{ position: absolute; top: 10px; left: 15px; color: #f8fafc; font-size: 13px; z-index: 10; pointer-events: none; }}
            .metric {{ font-weight: bold; color: {color}; }}
        </style>
    </head>
    <body>
        <div id="hud-container">
            <div id="overlay">
                <div>AI AUDIO FORENSIC HUD // 3D SPECTRAL MESH</div>
                <div>PREDICTION: <span class="metric">{prediction.upper()}</span></div>
                <div>SPOOF PROBABILITY: <span class="metric">{spoof_prob*100:.1f}%</span></div>
                <div>LATENCY: <span class="metric">{latency_ms:.1f} ms</span></div>
            </div>
            <div id="canvas-div"></div>
        </div>
        <script>
            const container = document.getElementById('hud-container');
            const scene = new THREE.Scene();
            scene.fog = new THREE.FogExp2(0x0b0f19, 0.02);
            const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(0, 15, 30);
            camera.lookAt(0, 0, 0);
            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);

            const gridHelper = new THREE.GridHelper(50, 50, 0x1e293b, 0x0f172a);
            scene.add(gridHelper);

            const width = 40, height = 30, segW = 40, segH = 30;
            const geometry = new THREE.PlaneGeometry(width, height, segW, segH);
            geometry.rotateX(-Math.PI / 2);
            const material = new THREE.MeshBasicMaterial({{
                color: {int(color.replace("#", "0x"), 16)},
                wireframe: true,
                transparent: true,
                opacity: 0.8
            }});
            const mesh = new THREE.Mesh(geometry, material);
            scene.add(mesh);

            let clock = new THREE.Clock();
            function animate() {{
                requestAnimationFrame(animate);
                let time = clock.getElapsedTime() * 2;
                const pos = geometry.attributes.position;
                for (let i = 0; i < pos.count; i++) {{
                    let x = pos.getX(i);
                    let z = pos.getZ(i);
                    let y = Math.sin(x * 0.3 + time) * Math.cos(z * 0.3 + time) * 3;
                    pos.setY(i, y);
                }}
                pos.needsUpdate = true;
                mesh.rotation.y = Math.sin(time * 0.1) * 0.2;
                renderer.render(scene, camera);
            }}
            animate();
            window.addEventListener('resize', () => {{
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=360)
