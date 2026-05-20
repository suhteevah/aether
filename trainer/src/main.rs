//! aether-train — minimal CLI driver for AetherLM training. Calls only into
//! `aether_rt` C-ABI symbols. Argument parsing is hand-rolled (no clap dep).

use std::path::PathBuf;
use trainer::config::{ModelConfig, TrainConfig};
use trainer::data::ByteDataset;
use trainer::model::{adamw_step, backward, clip_grads, forward, Model};
use trainer::rng::Rng;

#[derive(Debug)]
struct Cli {
    data: Option<PathBuf>,
    out: PathBuf,
    steps: usize,
    batch: usize,
    seq: usize,
    lr: f32,
    seed: u64,
    log_every: usize,
    config: String,
    synth_bytes: usize,
    world_size: usize,
}

fn parse_cli() -> Cli {
    let mut cli = Cli {
        data: None,
        out: PathBuf::from("checkpoints/aether_lm"),
        steps: 200,
        batch: 8,
        seq: 0,
        lr: 3e-3,
        seed: 42,
        log_every: 10,
        config: "nano".into(),
        synth_bytes: 64 * 1024,
        world_size: 1,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--data" => cli.data = Some(PathBuf::from(it.next().unwrap())),
            "--out"  => cli.out  = PathBuf::from(it.next().unwrap()),
            "--steps" => cli.steps = it.next().unwrap().parse().unwrap(),
            "--batch" => cli.batch = it.next().unwrap().parse().unwrap(),
            "--seq"   => cli.seq   = it.next().unwrap().parse().unwrap(),
            "--lr"    => cli.lr    = it.next().unwrap().parse().unwrap(),
            "--seed"  => cli.seed  = it.next().unwrap().parse().unwrap(),
            "--log-every" => cli.log_every = it.next().unwrap().parse().unwrap(),
            "--config" => cli.config = it.next().unwrap(),
            "--synth-bytes" => cli.synth_bytes = it.next().unwrap().parse().unwrap(),
            "--world-size" => cli.world_size = it.next().unwrap().parse().unwrap(),
            "-h" | "--help" => {
                eprintln!("aether-train [--data PATH] [--out PATH] [--config nano|tiny] [--steps N] [--batch N] [--seq N] [--lr F] [--seed N] [--log-every N] [--world-size N]");
                eprintln!("  --world-size N>1 enables data-parallel training across N GPUs via NCCL (requires --features nccl).");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {}", other);
                std::process::exit(2);
            }
        }
    }
    cli
}

fn main() {
    let cli = parse_cli();
    let mut cfg = match cli.config.as_str() {
        "nano" => ModelConfig::nano_cpu(),
        "tiny" => ModelConfig::tiny_3070ti(),
        other => { eprintln!("unknown --config {}", other); std::process::exit(2); }
    };
    if cli.seq > 0 { cfg.seq_len = cli.seq; }

    let train = TrainConfig {
        steps: cli.steps,
        batch_size: cli.batch,
        lr: cli.lr,
        seed: cli.seed,
        log_every: cli.log_every,
        ..TrainConfig::smoke()
    };

    eprintln!("[aether-train] config: {:?}", cfg);
    eprintln!("[aether-train] params: {} ({:.2}M)", cfg.num_params(), cfg.num_params() as f32 / 1e6);
    eprintln!("[aether-train] train: {:?}", train);

    let dataset = match cli.data {
        Some(p) => {
            eprintln!("[aether-train] loading corpus from {:?}", p);
            ByteDataset::from_file(&p, cfg.seq_len).expect("load dataset")
        }
        None => {
            eprintln!("[aether-train] using synthetic corpus ({} bytes)", cli.synth_bytes);
            ByteDataset::synthetic(cli.synth_bytes, cfg.seq_len)
        }
    };

    // Dispatch to the data-parallel loop when --world-size > 1.
    #[cfg(feature = "nccl")]
    if cli.world_size > 1 {
        eprintln!("[aether-train] data-parallel world_size={} (NCCL)", cli.world_size);
        let _trace = trainer::dp::train_dp(cfg.clone(), train.clone(), cli.world_size, dataset)
            .expect("dp training failed");
        eprintln!("[aether-train] dp training done");
        return;
    }
    #[cfg(not(feature = "nccl"))]
    if cli.world_size > 1 {
        eprintln!("[aether-train] --world-size > 1 requires building with --features nccl");
        std::process::exit(2);
    }

    let mut model = Model::new(cfg.clone(), train.seed);
    eprintln!("[aether-train] arena: {} f32 floats ({} MB)",
        model.n_params(), model.n_params() * 4 / (1024 * 1024));

    let mut rng = Rng::new(train.seed.wrapping_add(0xA17C));
    let t0 = std::time::Instant::now();
    let mut last_log = t0;
    let mut running = 0.0f64;
    let mut running_count = 0usize;

    for step in 0..train.steps {
        let lr = cosine_lr(step, train.steps, train.lr, train.warmup);
        let (ids, labels) = dataset.sample_batch(train.batch_size, &mut rng);
        let (act, loss) = forward(&model, &ids, &labels, train.batch_size);
        backward(&mut model, &act, &ids, &labels);
        let _norm = clip_grads(&mut model, train.grad_clip);
        adamw_step(&mut model, lr, 0.9, 0.95, 1e-8, train.weight_decay, (step + 1) as i64);

        running += loss as f64;
        running_count += 1;

        if step % train.log_every == 0 || step + 1 == train.steps {
            let now = std::time::Instant::now();
            let avg = running / running_count.max(1) as f64;
            let dt = now.duration_since(last_log).as_secs_f32();
            let toks = (train.batch_size * cfg.seq_len * train.log_every) as f32;
            let tps = if dt > 0.0 { toks / dt } else { 0.0 };
            eprintln!("[aether-train] step={:>5} loss={:.4} lr={:.2e} tok/s={:.0} elapsed={:.1}s",
                step, avg, lr, tps, t0.elapsed().as_secs_f32());
            last_log = now;
            running = 0.0;
            running_count = 0;
        }
    }

    eprintln!("[aether-train] done in {:.1}s", t0.elapsed().as_secs_f32());

    if let Some(parent) = cli.out.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    save_checkpoint(&cli.out, &model).expect("save");
    eprintln!("[aether-train] wrote {:?}.{{weights,meta}}", cli.out);
}

fn cosine_lr(step: usize, max_steps: usize, lr_max: f32, warmup: usize) -> f32 {
    if step < warmup { return lr_max * (step + 1) as f32 / warmup.max(1) as f32; }
    let p = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    lr_max * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
}

/// Custom checkpoint format: a header line + raw f32 LE for params.
/// Format: `b"AETHER01\n" + u64 LE param_count + raw f32 LE`.
/// The companion `.meta` file is JSON-ish (hand-rolled, no serde) with config.
///
/// The output stem is validated: must be within the current working
/// directory (after canonicalisation) and must not contain `..` segments,
/// to prevent a malicious CLI argument from overwriting arbitrary files.
fn save_checkpoint(stem: &std::path::Path, model: &Model) -> std::io::Result<()> {
    use std::io::Write;
    let stem = sanitize_output_stem(stem)?;
    let weights_path = stem.with_extension("weights");
    let meta_path = stem.with_extension("meta");
    let mut wf = std::fs::File::create(&weights_path)?;
    wf.write_all(b"AETHER01\n")?;
    let n = model.params.len() as u64;
    wf.write_all(&n.to_le_bytes())?;
    let bytes = unsafe { std::slice::from_raw_parts(model.params.as_ptr() as *const u8, model.params.len() * 4) };
    wf.write_all(bytes)?;

    let cfg = &model.cfg;
    let mut mf = std::fs::File::create(&meta_path)?;
    write!(mf,
        "{{\"vocab\":{},\"d_model\":{},\"n_layers\":{},\"n_heads\":{},\"d_ff\":{},\"seq_len\":{},\"params\":{}}}\n",
        cfg.vocab, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.seq_len, model.params.len(),
    )?;
    Ok(())
}

fn sanitize_output_stem(stem: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
    use std::io::{Error, ErrorKind};
    use std::path::Component;
    if stem.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(Error::new(ErrorKind::InvalidInput,
            "output path may not contain `..` segments"));
    }
    let cwd = std::env::current_dir()?.canonicalize()?;
    let parent = stem.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(parent)?;
    let parent_canon = parent.canonicalize()?;
    if !parent_canon.starts_with(&cwd) {
        return Err(Error::new(ErrorKind::InvalidInput,
            format!("output dir {:?} escapes cwd {:?}", parent_canon, cwd)));
    }
    let file_stem = stem.file_name().ok_or_else(|| Error::new(ErrorKind::InvalidInput,
        "output stem missing file name"))?;
    Ok(parent_canon.join(file_stem))
}
