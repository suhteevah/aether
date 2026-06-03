# Deploy — aether-vision (FR-V2)

Standing DINOv3 ViT-L/16 image-embedding service on the cnc P100 (GPU 1).
Coexists with the Qwen `aether-serve.service` (GPU 0, :18913) — never touch it.

**Live:** `cnc-server` / `http://192.168.168.100:18951` · `/v1/vision/embed` + `/health`.
Witnessed cosine 1.000000 vs the golden from cnc-localhost AND kokonoe-LAN.

## Build (clean FR-V1 tree, NOT /opt/aether)
```bash
# source synced to /root/aether-vit (current FR-V1 HEAD)
cd /root/aether-vit
export PATH=/root/.cargo/bin:$PATH
export LD_LIBRARY_PATH=/opt/visionsystem/runtime/cuda-extra/lib:/usr/local/cuda-12.8/lib64:/usr/local/lib
cargo build -p trainer --bin aether-serve --features cuda --release
install -m755 target/release/aether-serve /opt/aether-vision/bin/aether-serve
# weights (366 raw-f32 .bin tensors) live at /opt/aether-vision/wclean
```

## Install / run
```bash
cp aether-vision.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now aether-vision.service
curl -s http://127.0.0.1:18951/health   # -> ok
```

## GPU coordination (BINDING — openclaw `main`)
Standing ~1.45 GB GPU-1 tenant, granted + ledgered by `main` 2026-06-03.
**Yield for both-card matt-voice/trw-voice windows:** `systemctl stop
aether-vision` (SIGTERM → process exit → VRAM released in seconds); `systemctl
start aether-vision` after the window. Until `main`'s eviction trap learns about
:18951, self-yield immediately when `main` announces a both-card window.

## Firewall
`:18951` is unauthenticated; firewalld opened **only** for `192.168.168.0/24`
(LAN) + `100.64.0.0/10` (tailscale CGNAT). The tailscale *interface* may sit in a
different firewalld zone — add a zone-scoped rule there if the tailscale path is
needed (the LAN path is live).

## Client
visionsystem `AetherBackend::embed()` is the HTTP client; set
`AETHER_VISION_URL=http://192.168.168.100:18951`.
