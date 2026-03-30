#!/bin/bash
# ================================================================
# MOCHA 14B - RTX 6000 Ada (48GB) - DEFINITIVO
# Cola no terminal do RunPod e vai embora tomar café
# ================================================================
# PRÉ-REQUISITO ÚNICO: na criação do pod, expor porta 8188
# GPU: RTX 6000 Ada | Volume: 100GB
# ================================================================

set -e
cd /workspace

echo "=========================================="
echo " MoCha 14B - RTX 6000 Ada 48GB"
echo " Setup 100% automático"
echo "=========================================="

# ── 1. LIMPAR ────────────────────────────────────────────────────
echo "[1/8] Limpando..."
rm -rf /workspace/ComfyUI

# ── 2. COMFYUI ───────────────────────────────────────────────────
echo "[2/8] ComfyUI..."
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
pip install huggingface_hub[cli] soundfile

# ── 3. CUSTOM NODES ──────────────────────────────────────────────
echo "[3/8] Custom nodes..."
cd custom_nodes
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
cd ComfyUI-WanVideoWrapper && pip install -r requirements.txt && cd ..
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
cd ComfyUI-VideoHelperSuite && pip install -r requirements.txt && cd ..
git clone https://github.com/kijai/ComfyUI-KJNodes.git
cd ComfyUI-KJNodes && pip install -r requirements.txt && cd ..
git clone https://github.com/kijai/ComfyUI-segment-anything-2.git
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
git clone https://github.com/jamesWalker55/comfyui-various.git
cd /workspace/ComfyUI

# ── 4. FIXES PYTORCH 2.4 ────────────────────────────────────────
echo "[4/8] Fixes..."
LOADER="custom_nodes/ComfyUI-WanVideoWrapper/nodes_model_loading.py"
sed -i 's/raise ValueError("torch.backends.cuda.matmul.allow_fp16_accumulation/pass #raise ValueError("torch.backends.cuda.matmul.allow_fp16_accumulation/' "$LOADER"
sed -i "s/raise ValueError(f\"Can't import SageAttention/pass #raise ValueError(f\"Can't import SageAttention/" "$LOADER"

# ── 5. MODELOS ───────────────────────────────────────────────────
echo "[5/8] Baixando modelos (~27 GB)..."
python -c "
from huggingface_hub import hf_hub_download
import os, shutil

print('>>> [1/5] MoCha FP8 (14.3 GB)...')
hf_hub_download('Kijai/WanVideo_comfy_fp8_scaled',
    filename='MoCha/Wan2_1_mocha-14B-preview_fp8_e4m3fn_scaled_KJ.safetensors',
    local_dir='models/diffusion_models/WanVideo')
print('=== OK ===')

print('>>> [2/5] Text Encoder (11.4 GB)...')
hf_hub_download('Kijai/WanVideo_comfy',
    filename='umt5-xxl-enc-bf16.safetensors',
    local_dir='models/text_encoders/')
print('=== OK ===')

print('>>> [3/5] VAE (254 MB)...')
hf_hub_download('Kijai/WanVideo_comfy',
    filename='Wan2_1_VAE_bf16.safetensors',
    local_dir='models/vae/')
print('=== OK ===')

print('>>> [4/5] CLIP Vision (1.26 GB)...')
hf_hub_download('Comfy-Org/Wan_2.1_ComfyUI_repackaged',
    filename='split_files/clip_vision/clip_vision_h.safetensors',
    local_dir='models/clip_vision/')
src = 'models/clip_vision/split_files/clip_vision/clip_vision_h.safetensors'
if os.path.exists(src):
    shutil.move(src, 'models/clip_vision/clip_vision_h.safetensors')
    shutil.rmtree('models/clip_vision/split_files', ignore_errors=True)
print('=== OK ===')

print('>>> [5/5] LoRA Lightx2v T2V (631 MB)...')
hf_hub_download('Kijai/WanVideo_comfy',
    filename='Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors',
    local_dir='models/loras/')
print('=== OK ===')
print('=== TODOS OS MODELOS OK ===')
"

# ── 6. CORRIGIR WORKFLOW (TODAS AS CORREÇÕES) ────────────────────
echo "[6/8] Corrigindo workflow com TODAS as otimizações..."
WF="custom_nodes/ComfyUI-WanVideoWrapper/example_workflows/wanvideo_2_1_14B_MoCha_replace_subject_KJ_02.json"

python -c "
import json

f = '$WF'
t = open(f,'rb').read()

# === CAMINHOS: Windows → Linux ===
t = t.replace(b'WanVideo\x5c\x5cmocha\x5c\x5cMoCha\x5c\x5c', b'WanVideo/MoCha/')
t = t.replace(b'WanVideo\x5c\x5cLightx2v\x5c\x5c', b'Lightx2v/')
t = t.replace(b'wanvideo\x5c\x5cWan2_1_VAE', b'Wan2_1_VAE')
t = t.replace(b'rank64_bf16_.safetensors', b'rank64_bf16.safetensors')

# === sageattn → sdpa ===
t = t.replace(b'\"sageattn\"', b'\"sdpa\"')

open(f,'wb').write(t)

# === OTIMIZAÇÕES POR NÓ ===
data = json.load(open(f))
for node in data['nodes']:
    nid = node['id']
    vals = node.get('widgets_values', [])
    if not vals: continue

    # [128] Load Video: format wan, force_rate 0 (auto), cap 81
    if nid == 128 and isinstance(vals, dict):
        vals['format'] = 'wan'
        vals['force_rate'] = 0
        vals['frame_load_cap'] = 81
        print(f'[128] LoadVideo: wan, auto-fps, cap=81')

    # [302] MochaEmbeds: tiled_vae ON
    if nid == 302 and len(vals) >= 2:
        vals[1] = True
        print(f'[302] MochaEmbeds: tiled_vae=ON')

    # [304] VAE Decode: tiling ON
    if nid == 304:
        vals[0] = True
        print(f'[304] VAE Decode: tiling=ON')

    # [311] ModelLoader: main_device (48GB cabe!)
    if nid == 311:
        vals[3] = 'main_device'
        print(f'[311] ModelLoader: main_device')

    # [313] TextEncode: GPU (não CPU!)
    if nid == 313 and len(vals) >= 7:
        vals[6] = 'gpu'
        print(f'[313] TextEncode: gpu')

    # [314] Sampler: steps 6, cfg 1.0 (config que funcionou)
    if nid == 314:
        vals[0] = 6
        vals[1] = 1.0
        print(f'[314] Sampler: steps=6, cfg=1.0')

    # [325] SAM2: VIDEO (não single_image!)
    if nid == 325:
        vals[1] = 'video'
        print(f'[325] SAM2: video (CRÍTICO)')

    # [343] BlockSwap: 0 (48GB não precisa!)
    if nid == 343:
        vals[0] = 0
        print(f'[343] BlockSwap: 0 (48GB)')

    # [345] TorchCompile: inductor
    if nid == 345:
        vals[0] = 'inductor'
        vals[1] = False
        print(f'[345] TorchCompile: inductor')

    # [347] GrowMask: suavização adequada
    if nid == 347:
        vals[0] = 4     # expand
        vals[4] = 3      # blur
        vals[5] = 0.6    # lerp
        vals[6] = 0.6    # decay
        print(f'[347] GrowMask: expand=4, blur=3, lerp=0.6, decay=0.6')

json.dump(data, open(f,'w'), indent=2)

# Verificação
import re
t2 = open(f).read()
erros = []
if t2.count('\\\\\\\\') > 0: erros.append('Barras Windows')
if 'sageattn' in t2: erros.append('sageattn')
if 'single_image' in t2: erros.append('SAM2 single_image')
if erros:
    print(f'PROBLEMAS: {erros}')
else:
    print()
    print('=== WORKFLOW OK - ZERO ERROS ===')
"

mkdir -p user/default/workflows
cp "$WF" user/default/workflows/

# ── 7. FFMPEG ────────────────────────────────────────────────────
echo "[7/8] ffmpeg..."
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1

# ── 8. INICIAR ───────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " PRONTO! Acesse porta 8188"
echo ""
echo " Ctrl+O → wanvideo_2_1_14B_MoCha..."
echo "   1. Carregue vídeo"
echo "   2. Carregue imagem de referência"
echo "   3. Marque o sujeito no Points Editor"
echo "   4. Escreva prompt descritivo"
echo "   5. Run"
echo ""
echo " Tudo já configurado:"
echo "   SAM2: video (rastreia máscara)"
echo "   BlockSwap: 0 (GPU 100%)"
echo "   ModelLoader: main_device"
echo "   TextEncode: GPU"
echo "   Steps: 6 | CFG: 1.0 | sdpa"
echo "   LoRA: Lightx2v | GrowMask: suave"
echo "   FPS: auto-detecta do vídeo"
echo "mkdir -p /workspace/videos_salvos"
echo ""
echo "   LoRA: Lightx2v | GrowMask: suave"
echo "   FPS: auto-detecta do vídeo"
echo ""
echo " Dividir vídeo: ffmpeg -i input.mp4 \\"
echo "   -c copy -segment_time 3 -f segment \\"
echo "   -reset_timestamps 1 parte_%03d.mp4"
echo "=========================================="
echo ""

# O comando mkdir precisa estar solto (fora do echo) para ser executado
mkdir -p /workspace/videos_salvos

python main.py --listen 0.0.0.0 --port 8188 --output-directory /workspace/videos_salvos
