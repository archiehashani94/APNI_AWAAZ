"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { audioLevel } from "@/lib/audioMonitor";

type Mood = "idle" | "speaking" | "listening";

const NOISE = /* glsl */ `
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

  float snoise(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);

    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;

    i = mod289(i);
    vec4 p = permute(permute(permute(
              i.z + vec4(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0));

    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);

    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);

    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);

    vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
  }
`;

/**
 * The voice core: a red wireframe "anomaly" that reacts to the cloned audio.
 * Drag it with a mouse to spin it; it eases back into its own orbit.
 */
export default function AnomalyOrb({ mood }: { mood: Mood }) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const moodRef = useRef<Mood>(mood);

  useEffect(() => {
    moodRef.current = mood;
  }, [mood]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
    } catch {
      // A CSS glow inside the frame keeps the layout intact without WebGL.
      return;
    }

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    host.appendChild(renderer.domElement);
    renderer.domElement.className = "orb-canvas";
    renderer.domElement.setAttribute("aria-hidden", "true");

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 120);
    camera.position.set(0, 0, 9);

    const group = new THREE.Group();
    scene.add(group);

    const red = new THREE.Color(0xff4e42);

    const outerGeometry = new THREE.IcosahedronGeometry(2, 3);
    const outerMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uAudio: { value: 0 },
        uDistortion: { value: 0.6 },
        uColor: { value: red },
      },
      vertexShader: /* glsl */ `
        uniform float uTime;
        uniform float uAudio;
        uniform float uDistortion;
        varying vec3 vNormal;
        varying vec3 vPos;
        ${NOISE}

        void main() {
          vNormal = normalize(mat3(modelMatrix) * normal);

          float slow = uTime * 0.35;
          float n = snoise(vec3(position.x * 0.45, position.y * 0.45, position.z * 0.45 + slow));
          float n2 = snoise(vec3(position.x * 1.15 + 9.0, position.y * 1.15, position.z * 1.15 - slow));

          vec3 pos = position + normal * (n * 0.22 + n2 * 0.045) * uDistortion * (1.0 + uAudio * 1.5);
          vPos = (modelMatrix * vec4(pos, 1.0)).xyz;
          gl_Position = projectionMatrix * viewMatrix * vec4(vPos, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uTime;
        uniform float uAudio;
        uniform vec3 uColor;
        varying vec3 vNormal;
        varying vec3 vPos;

        void main() {
          vec3 viewDir = normalize(cameraPosition - vPos);
          float fresnel = pow(1.0 - max(dot(viewDir, normalize(vNormal)), 0.0), 1.5);
          float pulse = 0.74 + 0.26 * sin(uTime * 1.9);

          vec3 color = uColor * (0.38 + fresnel * 1.1) * pulse * (1.0 + uAudio * 1.3);
          float alpha = 0.3 + fresnel * 0.65;
          gl_FragColor = vec4(color, alpha);
        }
      `,
      wireframe: true,
      transparent: true,
      depthWrite: false,
    });
    const core = new THREE.Mesh(outerGeometry, outerMaterial);
    core.renderOrder = 2;
    group.add(core);

    const glowGeometry = new THREE.SphereGeometry(2.34, 48, 48);
    const glowMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uAudio: { value: 0 },
        uColor: { value: red },
      },
      vertexShader: /* glsl */ `
        uniform float uAudio;
        varying vec3 vNormal;
        varying vec3 vPos;

        void main() {
          vNormal = normalize(mat3(modelMatrix) * normal);
          vec3 pos = position * (1.0 + uAudio * 0.12);
          vPos = (modelMatrix * vec4(pos, 1.0)).xyz;
          gl_Position = projectionMatrix * viewMatrix * vec4(vPos, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uTime;
        uniform float uAudio;
        uniform vec3 uColor;
        varying vec3 vNormal;
        varying vec3 vPos;

        void main() {
          vec3 viewDir = normalize(cameraPosition - vPos);
          float fresnel = pow(1.0 - max(dot(viewDir, normalize(vNormal)), 0.0), 2.6 + uAudio * 2.0);
          float pulse = 0.72 + 0.28 * sin(uTime * 2.1);
          vec3 color = uColor * fresnel * pulse * (1.0 + uAudio * 2.2);
          gl_FragColor = vec4(color, fresnel * 0.45);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const glow = new THREE.Mesh(glowGeometry, glowMaterial);
    glow.renderOrder = 1;
    group.add(glow);

    // Sparse red dust, so the core sits in a field instead of a flat backdrop.
    const count = 1400;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const sizes = new Float32Array(count);
    const palette = [
      new THREE.Color(0xff4e42),
      new THREE.Color(0xc2362f),
      new THREE.Color(0xffb3ab),
    ];
    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 80;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 80;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 80;
      const color = palette[Math.floor(Math.random() * palette.length)];
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
      sizes[i] = Math.random() * 0.07 + 0.04;
    }
    const dustGeometry = new THREE.BufferGeometry();
    dustGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    dustGeometry.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
    dustGeometry.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
    const dustMaterial = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 } },
      vertexShader: /* glsl */ `
        attribute vec3 aColor;
        attribute float aSize;
        uniform float uTime;
        varying vec3 vColor;

        void main() {
          vColor = aColor;
          vec3 pos = position;
          pos.x += sin(uTime * 0.09 + position.z * 0.2) * 0.09;
          pos.y += cos(uTime * 0.09 + position.x * 0.2) * 0.09;
          pos.z += sin(uTime * 0.09 + position.y * 0.2) * 0.09;

          vec4 mv = modelViewMatrix * vec4(pos, 1.0);
          gl_PointSize = aSize * (330.0 / max(1.0, -mv.z));
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: /* glsl */ `
        varying vec3 vColor;

        void main() {
          float r = distance(gl_PointCoord, vec2(0.5, 0.5));
          if (r > 0.5) discard;
          float glow = pow(1.0 - r * 2.0, 2.0);
          gl_FragColor = vec4(vColor, glow);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const dust = new THREE.Points(dustGeometry, dustMaterial);
    scene.add(dust);

    const resize = () => {
      const width = host.clientWidth || 320;
      const height = host.clientHeight || 320;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    // Mouse drag spins the core. Touch is left alone so the page still scrolls.
    const canvas = renderer.domElement;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let spinX = 0;
    let spinY = 0;
    let velX = 0;
    let velY = 0;

    const onDown = (event: PointerEvent) => {
      if (event.pointerType !== "mouse") return;
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      try {
        canvas.setPointerCapture(event.pointerId);
      } catch {
        /* pointer capture is optional */
      }
    };
    const onMove = (event: PointerEvent) => {
      if (!dragging) return;
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      velY = dx * 0.006;
      velX = dy * 0.006;
      spinY += velY;
      spinX = Math.max(-0.9, Math.min(0.9, spinX + velX));
    };
    const onUp = (event: PointerEvent) => {
      dragging = false;
      try {
        canvas.releasePointerCapture(event.pointerId);
      } catch {
        /* already released */
      }
    };
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);

    let frame = 0;
    let smooth = 0;
    const clock = new THREE.Clock();

    const render = () => {
      const time = clock.getElapsedTime();
      const now = moodRef.current;

      let target = audioLevel();
      if (now === "speaking") {
        target = Math.max(target, 0.34 + 0.16 * Math.sin(time * 8.5) + 0.08 * Math.sin(time * 21.3 + 1.4));
      } else if (now === "listening") {
        target = Math.max(target, 0.13 + 0.07 * Math.sin(time * 3.2) + 0.04 * Math.sin(time * 7.7));
      } else {
        target = Math.max(target, 0.035 + 0.02 * Math.sin(time * 1.3));
      }
      smooth += (target - smooth) * 0.12;

      if (!dragging) {
        spinX = Math.max(-0.9, Math.min(0.9, spinX + velX));
        spinY += velY;
        velX *= 0.94;
        velY *= 0.94;
      }

      group.rotation.y = time * 0.22 + spinY;
      group.rotation.x = Math.sin(time * 0.16) * 0.16 + spinX;
      group.rotation.z = Math.sin(time * 0.09) * 0.08;

      camera.position.x = Math.sin(time * 0.23) * 0.25;
      camera.position.y = Math.cos(time * 0.19) * 0.15;
      camera.lookAt(0, 0, 0);

      outerMaterial.uniforms.uTime.value = time;
      outerMaterial.uniforms.uAudio.value = smooth;
      glowMaterial.uniforms.uTime.value = time;
      glowMaterial.uniforms.uAudio.value = smooth;
      dustMaterial.uniforms.uTime.value = time;

      dust.rotation.y = time * 0.008;
      dust.rotation.x = time * 0.004;

      renderer.render(scene, camera);
      frame = requestAnimationFrame(render);
    };

    if (reduced) {
      group.rotation.set(0.3, 0.6, 0);
      outerMaterial.uniforms.uTime.value = 0.9;
      glowMaterial.uniforms.uTime.value = 0.9;
      renderer.render(scene, camera);
    } else {
      frame = requestAnimationFrame(render);
    }

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      outerGeometry.dispose();
      outerMaterial.dispose();
      glowGeometry.dispose();
      glowMaterial.dispose();
      dustGeometry.dispose();
      dustMaterial.dispose();
      renderer.dispose();
      if (canvas.parentNode === host) host.removeChild(canvas);
    };
  }, []);

  return <div className="orb-host" ref={hostRef} />;
}
