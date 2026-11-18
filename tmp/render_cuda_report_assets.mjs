import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const sharp = require('sharp');

const root = process.cwd();
const figuresDir = path.join(root, 'figures');
const mermaidBundle = path.join(root, 'llm-provider-comparison', '_shared', 'js', 'mermaid.min.js');
const chromePath = 'C:/Program Files/Google/Chrome/Application/chrome.exe';

async function renderMermaid(fileName, index) {
  const mmdPath = path.join(figuresDir, fileName);
  const code = await fs.readFile(mmdPath, 'utf8');
  const browser = await chromium.launch({
    headless: true,
    executablePath: chromePath,
    args: ['--no-sandbox'],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1800, height: 1200 } });
    await page.setContent('<html><body style="margin:0;background:#ffffff"></body></html>');
    await page.addScriptTag({ content: await fs.readFile(mermaidBundle, 'utf8') });
    const svg = await page.evaluate(async ({ source, graphId }) => {
      mermaid.initialize({
        startOnLoad: false,
        theme: 'base',
        securityLevel: 'loose',
        flowchart: { htmlLabels: true, useMaxWidth: false, curve: 'basis' },
        themeVariables: { fontFamily: 'Arial, Microsoft YaHei, sans-serif', fontSize: '18px' },
      });
      const result = await mermaid.render(graphId, source);
      return result.svg;
    }, { source: code, graphId: `cuda_report_${index}` });
    const stem = fileName.replace(/\.mmd$/i, '');
    const svgPath = path.join(figuresDir, `${stem}.svg`);
    const pngPath = path.join(figuresDir, `${stem}.png`);
    await fs.writeFile(svgPath, svg, 'utf8');
    // Mermaid uses foreignObject for HTML labels. Chromium renders those
    // labels faithfully; rasterizing the raw SVG with librsvg would drop them.
    const viewBox = (svg.match(/viewBox="([^"]+)"/) || [])[1] || '0 0 1600 900';
    const [, , vbW, vbH] = viewBox.split(/\s+/).map(Number);
    await page.setViewportSize({ width: Math.min(2400, Math.max(900, Math.ceil(vbW + 40))), height: Math.min(1800, Math.max(500, Math.ceil(vbH + 40))) });
    await page.setContent(`<html><body style="margin:0;background:#ffffff"><div id="canvas" style="display:inline-block;background:#ffffff;padding:20px">${svg}</div></body></html>`);
    await page.locator('#canvas').screenshot({ path: pngPath, scale: 'device' });
    console.log(`rendered ${fileName} -> ${path.basename(svgPath)}, ${path.basename(pngPath)}`);
  } finally {
    await browser.close();
  }
}

function esc(value) {
  return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

async function makeBenchmarkFigure() {
  const inputDir = path.join(root, 'artifacts', 'cuda_stage1_probe');
  const files = [
    ['固定输入算子微基准', 'optimized_cuda_gpu_two_way.json', 'mean_ms', 'ms'],
    ['MNIST 前向 + 反向', 'optimized_mnist_gpu_two_way.json', 'mean_batch_ms', 'ms'],
    ['MNIST Adam 训练冒烟', 'optimized_mnist_training_gpu_two_way.json', 'epoch_ms', 'ms'],
  ];
  const rows = [];
  for (const [label, file, key, unit] of files) {
    const data = JSON.parse(await fs.readFile(path.join(inputDir, file), 'utf8'));
    const r = data.results;
    const cupy = key === 'epoch_ms' ? r.cupy[key][0] : r.cupy[key];
    const raw = key === 'epoch_ms' ? r.rawmodule[key][0] : r.rawmodule[key];
    rows.push({ label, cupy, raw, speedup: cupy / raw, unit });
  }
  const width = 1600;
  const height = 900;
  const left = 420;
  const right = 130;
  const top = 130;
  const bottom = 130;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const maxVal = Math.max(...rows.flatMap(r => [r.cupy, r.raw])) * 1.18;
  const x = value => left + (value / maxVal) * plotW;
  const rowGap = plotH / rows.length;
  const barH = 42;
  let body = '';
  body += `<rect width="${width}" height="${height}" fill="#ffffff"/>`;
  body += `<text x="${left}" y="58" font-family="Microsoft YaHei, Arial, sans-serif" font-size="32" font-weight="700" fill="#111827">CUDA C RawModule 与 CuPy：当前实测耗时</text>`;
  body += `<text x="${left}" y="94" font-family="Microsoft YaHei, Arial, sans-serif" font-size="20" fill="#4B5563">同一设备、固定小规模输入；数值越短越快，右侧标注为 RawModule 相对 CuPy 的加速倍数</text>`;
  for (let i = 0; i <= 4; i++) {
    const value = maxVal * i / 4;
    const xx = x(value);
    body += `<line x1="${xx}" y1="${top}" x2="${xx}" y2="${height - bottom}" stroke="#E5E7EB" stroke-width="2"/>`;
    body += `<text x="${xx}" y="${height - bottom + 38}" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" fill="#6B7280">${value.toFixed(0)} ms</text>`;
  }
  rows.forEach((r, i) => {
    const center = top + rowGap * i + rowGap / 2;
    const cupyY = center - barH - 8;
    const rawY = center + 8;
    body += `<text x="${left - 28}" y="${center + 8}" text-anchor="end" font-family="Microsoft YaHei, Arial, sans-serif" font-size="22" font-weight="600" fill="#111827">${esc(r.label)}</text>`;
    body += `<rect x="${left}" y="${cupyY}" width="${Math.max(1, x(r.cupy) - left)}" height="${barH}" rx="8" fill="#9CA3AF"/>`;
    body += `<rect x="${left}" y="${rawY}" width="${Math.max(1, x(r.raw) - left)}" height="${barH}" rx="8" fill="#2563EB"/>`;
    body += `<text x="${x(r.cupy) + 12}" y="${cupyY + 30}" font-family="Arial, sans-serif" font-size="20" fill="#374151">CuPy ${r.cupy.toFixed(2)} ms</text>`;
    body += `<text x="${x(r.raw) + 12}" y="${rawY + 30}" font-family="Arial, sans-serif" font-size="20" fill="#1D4ED8">RawModule ${r.raw.toFixed(2)} ms</text>`;
    body += `<text x="${width - right + 10}" y="${center + 8}" text-anchor="end" font-family="Arial, sans-serif" font-size="28" font-weight="700" fill="#111827">${r.speedup.toFixed(2)}×</text>`;
  });
  body += `<rect x="${left}" y="${height - 72}" width="24" height="24" rx="4" fill="#9CA3AF"/><text x="${left + 36}" y="${height - 52}" font-family="Microsoft YaHei, Arial, sans-serif" font-size="19" fill="#374151">CuPy</text>`;
  body += `<rect x="${left + 140}" y="${height - 72}" width="24" height="24" rx="4" fill="#2563EB"/><text x="${left + 176}" y="${height - 52}" font-family="Microsoft YaHei, Arial, sans-serif" font-size="19" fill="#374151">CUDA C RawModule</text>`;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${body}</svg>`;
  const outDir = path.join(root, 'artifacts', 'cuda_report_figures');
  await fs.mkdir(outDir, { recursive: true });
  await fs.writeFile(path.join(outDir, 'cuda-speedup-summary.svg'), svg, 'utf8');
  await sharp(Buffer.from(svg)).png({ compressionLevel: 9 }).toFile(path.join(outDir, 'cuda-speedup-summary.png'));
  console.log('rendered benchmark summary');
}

await renderMermaid('cuda-c-backend-route.mmd', 1);
await renderMermaid('cuda-operator-dataflow.mmd', 2);
await renderMermaid('cuda-stage-roadmap.mmd', 3);
await makeBenchmarkFigure();
