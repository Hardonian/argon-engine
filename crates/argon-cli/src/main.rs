//! Reproducible run, benchmark, inspect, and replay CLI for Argon Engine.

#![forbid(non_ascii_idents)]

use argon_backend::{
    AcceleratedCapabilities, CudaCapabilities, LocalAiStack, LocalHardware, LocalStackKind,
};
use argon_ffi::ARGON_ABI_VERSION;
use argon_kernel::{PersistentGridConfig, PlasticityConfig};
use argon_medium::{CoupledOscillatorMedium, MediumConfig};
use argon_runtime::{ArgonCoordinator, ControlFrame};
use std::env;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_ITERATIONS: usize = 10_000;
const DEFAULT_WARMUP: usize = 1_000;
const TILE_DIM: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Command {
    Bench,
    Run,
    Inspect,
    Doctor,
    Models,
    Complete,
    Pull,
    Unload,
    ModelInfo,
    Replay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Text,
    Json,
    Csv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scenario {
    Equilibrium,
    Shock,
    Sustained,
}

#[derive(Debug)]
struct Options {
    command: Command,
    medium: bool,
    iterations: usize,
    warmup: usize,
    format: OutputFormat,
    scenario: Scenario,
    record: Option<PathBuf>,
    replay: Option<PathBuf>,
    local_stack: LocalStackKind,
    endpoint: Option<String>,
    allow_non_loopback: bool,
    model: Option<String>,
    prompt: Option<String>,
    max_tokens: u32,
    stream: bool,
    api_key: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct Report {
    iterations: usize,
    p50: u64,
    p95: u64,
    p99: u64,
    final_energy: f32,
    weight_checksum: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments.iter().any(|argument| argument == "--help") {
        print_help();
        return Ok(());
    }
    let options = parse_options(&arguments)?;
    match options.command {
        Command::Inspect => print_inspection(options.format),
        Command::Doctor => doctor(&options)?,
        Command::Models => list_models(&options)?,
        Command::Complete => complete(&options)?,
        Command::Pull => pull(&options)?,
        Command::Unload => unload(&options)?,
        Command::ModelInfo => model_info(&options)?,
        Command::Replay => replay(&options)?,
        Command::Bench | Command::Run => {
            let report = execute(&options)?;
            print_report(&options, report);
            if let Some(path) = &options.record {
                write_recording(path, &options, report)?;
            }
        }
    }
    Ok(())
}

fn print_help() {
    println!(
        "Usage: argon-cli [bench|run|inspect|doctor|models|complete|pull|unload|model-info|replay FILE] [--medium] \
         [--iterations N] [--warmup N] [--scenario equilibrium|shock|sustained] \
         [--format text|json|csv] [--record FILE] [--local-stack auto|ollama|llama-cpp|lm-studio|openai-compatible] \
         [--endpoint http://127.0.0.1:PORT] [--model MODEL] [--prompt TEXT] [--max-tokens N] \
         [--stream] [--api-key KEY] [--allow-non-loopback] [--config FILE]"
    );
}

fn print_inspection(format: OutputFormat) {
    let cuda = CudaCapabilities::probe();
    let hardware = LocalHardware::probe();
    let accelerated = AcceleratedCapabilities::probe();
    let cuda_devices = cuda
        .as_ref()
        .map_or(0, |capabilities| capabilities.devices.len());
    let local_stack = LocalAiStack::discover();
    match format {
        OutputFormat::Text => {
            println!("argon_version={} abi_version={} cpu=true wasm=true cuda_devices={cuda_devices} cuda_runtime=false local_stack={}", env!("CARGO_PKG_VERSION"), ARGON_ABI_VERSION, local_stack.as_ref().map_or("none", |stack| match stack.kind { LocalStackKind::Ollama => "ollama", LocalStackKind::LlamaCpp => "llama-cpp", LocalStackKind::LmStudio => "lm-studio", LocalStackKind::OpenAiCompatible => "openai-compatible", LocalStackKind::Auto => "auto" }));
            println!(
                "accelerated vulkan={} metal={} rocm={} platforms=[{}]",
                accelerated.vulkan_available,
                accelerated.metal_available,
                accelerated.rocm_available,
                accelerated.detected_platforms.join(", ")
            );
            for gpu in &hardware.gpus {
                println!(
                    "gpu name={} vram_mib={} tier={:?} guidance={}",
                    gpu.name,
                    gpu.memory_mib
                        .map_or_else(|| "unknown".to_owned(), |memory| memory.to_string()),
                    gpu.tier,
                    gpu.tier.guidance()
                );
            }
        }
        OutputFormat::Json => {
            println!("{{\"argon_version\":\"{}\",\"abi_version\":{},\"cpu\":true,\"wasm\":true,\"cuda_devices\":{},\"cuda_runtime\":false,\"local_stack\":{},\"accelerated\":{{\"vulkan\":{},\"metal\":{},\"rocm\":{},\"platforms\":{:?}}},\"gpus\":[{}]}}", env!("CARGO_PKG_VERSION"), ARGON_ABI_VERSION, cuda_devices, local_stack.as_ref().map_or_else(|| "null".to_owned(), |stack| format!("\"{}\"", stack.endpoint.url())), accelerated.vulkan_available, accelerated.metal_available, accelerated.rocm_available, accelerated.detected_platforms, hardware.gpus.iter().map(|gpu| format!("{{\"name\":\"{}\",\"vram_mib\":{},\"tier\":\"{:?}\"}}", gpu.name.replace('"', "\\\""), gpu.memory_mib.map_or_else(|| "null".to_owned(), |memory| memory.to_string()), gpu.tier)).collect::<Vec<_>>().join(","));
        }
        OutputFormat::Csv => {
            println!("name,vram_mib,tier,guidance");
            for gpu in &hardware.gpus {
                println!(
                    "\"{}\",{},\"{:?}\",\"{}\"",
                    gpu.name.replace('"', "\"\""),
                    gpu.memory_mib
                        .map_or_else(String::new, |memory| memory.to_string()),
                    gpu.tier,
                    gpu.tier.guidance()
                );
            }
            println!("platform,available");
            println!("vulkan,{}", accelerated.vulkan_available);
            println!("metal,{}", accelerated.metal_available);
            println!("rocm,{}", accelerated.rocm_available);
        }
    }
}

fn doctor(options: &Options) -> Result<(), Box<dyn Error>> {
    let stack = selected_stack(options)?;
    stack.health_check()?;
    match options.format {
        OutputFormat::Text => println!(
            "local_stack=healthy kind={:?} endpoint={}",
            stack.kind,
            stack.endpoint.url()
        ),
        OutputFormat::Json => println!(
            "{{\"local_stack\":\"healthy\",\"kind\":\"{:?}\",\"endpoint\":\"{}\"}}",
            stack.kind,
            stack.endpoint.url()
        ),
        OutputFormat::Csv => {
            println!("status,kind,endpoint");
            println!("healthy,{:?},{}", stack.kind, stack.endpoint.url());
        }
    }
    Ok(())
}

fn list_models(options: &Options) -> Result<(), Box<dyn Error>> {
    let models = selected_stack(options)?.list_models()?;
    match options.format {
        OutputFormat::Text => {
            for model in &models {
                println!("{model}");
            }
        }
        OutputFormat::Json => println!("{models:?}"),
        OutputFormat::Csv => {
            println!("model");
            for model in &models {
                println!("\"{}\"", model.replace('"', "\"\""));
            }
        }
    }
    Ok(())
}

fn complete(options: &Options) -> Result<(), Box<dyn Error>> {
    let model = options
        .model
        .as_deref()
        .ok_or("complete requires --model MODEL")?;
    let prompt = options
        .prompt
        .as_deref()
        .ok_or("complete requires --prompt TEXT")?;
    let stack = selected_stack(options)?;
    if options.stream {
        let response = stack.complete_stream(model, prompt, options.max_tokens, |chunk| {
            if options.format == OutputFormat::Text {
                print!("{chunk}");
                let _ = std::io::stdout().flush();
            }
        })?;
        match options.format {
            OutputFormat::Text => println!(),
            OutputFormat::Json => println!("{{\"response\":{response:?}}}"),
            OutputFormat::Csv => {
                println!("response");
                println!("\"{}\"", response.replace('"', "\"\""));
            }
        }
    } else {
        let response = stack.complete(model, prompt, options.max_tokens)?;
        match options.format {
            OutputFormat::Text => println!("{response}"),
            OutputFormat::Json => println!("{{\"response\":{response:?}}}"),
            OutputFormat::Csv => {
                println!("response");
                println!("\"{}\"", response.replace('"', "\"\""));
            }
        }
    }
    Ok(())
}

fn pull(options: &Options) -> Result<(), Box<dyn Error>> {
    let model = options
        .model
        .as_deref()
        .ok_or("pull requires --model MODEL")?;
    let stack = selected_stack(options)?;
    stack.pull_model(model)?;
    match options.format {
        OutputFormat::Text => println!("model={model} status=pulled"),
        OutputFormat::Json => println!("{{\"model\":\"{model}\",\"status\":\"pulled\"}}"),
        OutputFormat::Csv => println!("model,status\n\"{model}\",pulled"),
    }
    Ok(())
}

fn unload(options: &Options) -> Result<(), Box<dyn Error>> {
    let model = options
        .model
        .as_deref()
        .ok_or("unload requires --model MODEL")?;
    let stack = selected_stack(options)?;
    stack.unload_model(model)?;
    match options.format {
        OutputFormat::Text => println!("model={model} status=unloaded"),
        OutputFormat::Json => println!("{{\"model\":\"{model}\",\"status\":\"unloaded\"}}"),
        OutputFormat::Csv => println!("model,status\n\"{model}\",unloaded"),
    }
    Ok(())
}

fn model_info(options: &Options) -> Result<(), Box<dyn Error>> {
    let model = options
        .model
        .as_deref()
        .ok_or("model-info requires --model MODEL")?;
    let stack = selected_stack(options)?;
    let info = stack.model_info(model)?;
    match options.format {
        OutputFormat::Text => println!("model={model} info={info}"),
        OutputFormat::Json => println!("{info}"),
        OutputFormat::Csv => {
            println!("model,info");
            println!("\"{model}\",\"{}\"", info.to_string().replace('"', "\"\""));
        }
    }
    Ok(())
}

fn selected_stack(options: &Options) -> Result<LocalAiStack, Box<dyn Error>> {
    if options.local_stack == LocalStackKind::Auto {
        Ok(LocalAiStack::discover()
            .ok_or("no supported local model server responded on loopback")?)
    } else {
        Ok(LocalAiStack::new_authenticated(
            options.local_stack,
            options.endpoint.as_deref(),
            options.allow_non_loopback,
            options.api_key.clone(),
        )?)
    }
}

fn parse_options(arguments: &[String]) -> Result<Options, Box<dyn Error>> {
    let expanded = expand_config_arguments(arguments)?;

    let mut options = default_options();

    let mut index = 0;
    if let Some(command) = expanded.first().filter(|value| !value.starts_with('-')) {
        options.command = match command.as_str() {
            "bench" => Command::Bench,
            "run" => Command::Run,
            "inspect" => Command::Inspect,
            "doctor" => Command::Doctor,
            "models" => Command::Models,
            "complete" => Command::Complete,
            "pull" => Command::Pull,
            "unload" => Command::Unload,
            "model-info" | "info" => Command::ModelInfo,
            "replay" => Command::Replay,
            _ => return Err(format!("unknown command: {command}").into()),
        };
        index += 1;
        if options.command == Command::Replay {
            options.replay = Some(PathBuf::from(
                expanded
                    .get(index)
                    .ok_or("replay requires a recording path")?,
            ));
            index += 1;
        }
    }

    while index < expanded.len() {
        match expanded[index].as_str() {
            "--medium" => options.medium = true,
            "--iterations" => {
                index += 1;
                options.iterations = parse_positive(&expanded, index, "--iterations")?;
            }
            "--warmup" => {
                index += 1;
                options.warmup = parse_usize(&expanded, index, "--warmup")?;
            }
            "--format" => {
                index += 1;
                options.format = match value_at(&expanded, index, "--format")? {
                    "text" => OutputFormat::Text,
                    "json" => OutputFormat::Json,
                    "csv" => OutputFormat::Csv,
                    value => return Err(format!("unknown format: {value}").into()),
                };
            }
            "--scenario" => {
                index += 1;
                options.scenario = match value_at(&expanded, index, "--scenario")? {
                    "equilibrium" => Scenario::Equilibrium,
                    "shock" => Scenario::Shock,
                    "sustained" => Scenario::Sustained,
                    value => return Err(format!("unknown scenario: {value}").into()),
                };
                options.medium = true;
            }
            "--record" => {
                index += 1;
                options.record = Some(PathBuf::from(value_at(&expanded, index, "--record")?));
            }
            "--local-stack" => {
                index += 1;
                options.local_stack =
                    LocalStackKind::parse(value_at(&expanded, index, "--local-stack")?)?;
            }
            "--endpoint" => {
                index += 1;
                options.endpoint = Some(value_at(&expanded, index, "--endpoint")?.to_owned());
            }
            "--allow-non-loopback" => options.allow_non_loopback = true,
            "--model" => {
                index += 1;
                options.model = Some(value_at(&expanded, index, "--model")?.to_owned());
            }
            "--prompt" => {
                index += 1;
                options.prompt = Some(value_at(&expanded, index, "--prompt")?.to_owned());
            }
            "--max-tokens" => {
                index += 1;
                options.max_tokens =
                    u32::try_from(parse_positive(&expanded, index, "--max-tokens")?)?;
            }
            "--stream" => options.stream = true,
            "--api-key" => {
                index += 1;
                options.api_key = Some(value_at(&expanded, index, "--api-key")?.to_owned());
            }
            value => return Err(format!("unknown argument: {value}").into()),
        }
        index += 1;
    }
    Ok(options)
}

const fn default_options() -> Options {
    Options {
        command: Command::Bench,
        medium: false,
        iterations: DEFAULT_ITERATIONS,
        warmup: DEFAULT_WARMUP,
        format: OutputFormat::Text,
        scenario: Scenario::Equilibrium,
        record: None,
        replay: None,
        local_stack: LocalStackKind::Auto,
        endpoint: None,
        allow_non_loopback: false,
        model: None,
        prompt: None,
        max_tokens: 256,
        stream: false,
        api_key: None,
    }
}

fn expand_config_arguments(arguments: &[String]) -> Result<Vec<String>, Box<dyn Error>> {
    let mut expanded = arguments.to_vec();
    if let Some(index) = expanded.iter().position(|argument| argument == "--config") {
        let path = expanded.get(index + 1).ok_or("--config requires a path")?;
        let mut from_file = parse_config(Path::new(path))?;
        expanded.drain(index..=index + 1);
        from_file.extend(expanded);
        return Ok(from_file);
    }
    Ok(expanded)
}

fn parse_config(path: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    let mut arguments = Vec::new();
    for line in contents.lines() {
        let line = line.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or("config lines must use key=value")?;
        let key = key.trim().replace('_', "-");
        let value = value.trim().trim_matches('"');
        if key == "medium" {
            if value == "true" {
                arguments.push("--medium".to_owned());
            }
        } else {
            arguments.push(format!("--{key}"));
            arguments.push(value.to_owned());
        }
    }
    Ok(arguments)
}

fn parse_positive(arguments: &[String], index: usize, flag: &str) -> Result<usize, Box<dyn Error>> {
    let parsed = parse_usize(arguments, index, flag)?;
    if parsed == 0 {
        return Err(format!("{flag} must be greater than zero").into());
    }
    Ok(parsed)
}

fn parse_usize(arguments: &[String], index: usize, flag: &str) -> Result<usize, Box<dyn Error>> {
    Ok(value_at(arguments, index, flag)?.parse()?)
}

fn value_at<'a>(
    arguments: &'a [String],
    index: usize,
    flag: &str,
) -> Result<&'a str, Box<dyn Error>> {
    arguments
        .get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn execute(options: &Options) -> Result<Report, Box<dyn Error>> {
    let mut coordinator = coordinator()?;
    let mut samples = Vec::with_capacity(options.iterations);
    let mut final_energy = 0.0_f32;
    if options.medium {
        let mut medium = CoupledOscillatorMedium::new(MediumConfig::default())?;
        initialize_scenario(&mut medium, options.scenario)?;
        for iteration in 0..options.warmup.saturating_add(options.iterations) {
            if options.scenario == Scenario::Sustained && iteration % 250 == 0 {
                medium.perturb(iteration / 250 % 4, 0.05, 0.1)?;
            }
            let started = Instant::now();
            let telemetry = medium.step(&mut coordinator)?;
            if iteration >= options.warmup {
                samples.push(elapsed_nanos(started));
            }
            final_energy = telemetry.energy;
        }
    } else {
        let mut features = vec![0.25_f32; TILE_DIM];
        let mut output = vec![0.0_f32; TILE_DIM];
        for iteration in 0..options.warmup.saturating_add(options.iterations) {
            let started = Instant::now();
            coordinator.submit_frame(ControlFrame {
                sequence_id: u64::try_from(iteration)?,
                phase_polarity: 1.0,
                features: std::mem::take(&mut features),
            })?;
            let _telemetry = coordinator.poll_and_step_recycle(&mut output, &mut features)?;
            if iteration >= options.warmup {
                samples.push(elapsed_nanos(started));
            }
        }
    }
    samples.sort_unstable();
    Ok(Report {
        iterations: options.iterations,
        p50: percentile(&samples, 50),
        p95: percentile(&samples, 95),
        p99: percentile(&samples, 99),
        final_energy,
        weight_checksum: coordinator.weight_checksum(),
    })
}

fn initialize_scenario(
    medium: &mut CoupledOscillatorMedium,
    scenario: Scenario,
) -> Result<(), Box<dyn Error>> {
    match scenario {
        Scenario::Equilibrium => {}
        Scenario::Shock => medium.perturb(0, 1.4, 2.5)?,
        Scenario::Sustained => medium.perturb(0, 0.4, 0.5)?,
    }
    Ok(())
}

fn print_report(options: &Options, report: Report) {
    let mode = if options.medium {
        "medium"
    } else {
        "coordinator"
    };
    match options.format {
        OutputFormat::Text => println!(
            "mode={mode} iterations={} p50={}ns p95={}ns p99={}ns energy={} checksum={:016x}",
            report.iterations,
            report.p50,
            report.p95,
            report.p99,
            report.final_energy,
            report.weight_checksum
        ),
        OutputFormat::Json => println!(
            "{{\"mode\":\"{mode}\",\"iterations\":{},\"p50_ns\":{},\"p95_ns\":{},\"p99_ns\":{},\"final_energy\":{},\"weight_checksum\":\"{:016x}\"}}",
            report.iterations,
            report.p50,
            report.p95,
            report.p99,
            report.final_energy,
            report.weight_checksum
        ),
        OutputFormat::Csv => {
            println!("mode,iterations,p50_ns,p95_ns,p99_ns,final_energy,weight_checksum");
            println!(
                "{mode},{},{},{},{},{},{:016x}",
                report.iterations,
                report.p50,
                report.p95,
                report.p99,
                report.final_energy,
                report.weight_checksum
            );
        }
    }
}

fn write_recording(path: &Path, options: &Options, report: Report) -> Result<(), Box<dyn Error>> {
    let scenario = match options.scenario {
        Scenario::Equilibrium => "equilibrium",
        Scenario::Shock => "shock",
        Scenario::Sustained => "sustained",
    };
    let contents = format!(
        "version=1\nmedium={}\niterations={}\nwarmup={}\nscenario={scenario}\nfinal_energy_bits={}\nweight_checksum={}\n",
        options.medium,
        options.iterations,
        options.warmup,
        report.final_energy.to_bits(),
        report.weight_checksum
    );
    fs::write(path, contents)?;
    Ok(())
}

fn replay(options: &Options) -> Result<(), Box<dyn Error>> {
    let path = options.replay.as_ref().ok_or("missing replay path")?;
    let contents = fs::read_to_string(path)?;
    let read = |key: &str| -> Result<&str, Box<dyn Error>> {
        contents
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{key}=")))
            .ok_or_else(|| format!("recording lacks {key}").into())
    };
    if read("version")? != "1" {
        return Err("unsupported recording version".into());
    }
    let mut replay_options = Options {
        command: Command::Run,
        medium: read("medium")?.parse()?,
        iterations: read("iterations")?.parse()?,
        warmup: read("warmup")?.parse()?,
        format: options.format,
        scenario: match read("scenario")? {
            "equilibrium" => Scenario::Equilibrium,
            "shock" => Scenario::Shock,
            "sustained" => Scenario::Sustained,
            _ => return Err("recording has an unknown scenario".into()),
        },
        record: None,
        replay: None,
        local_stack: LocalStackKind::Auto,
        endpoint: None,
        allow_non_loopback: false,
        model: None,
        prompt: None,
        max_tokens: 256,
        stream: false,
        api_key: None,
    };
    replay_options.format = OutputFormat::Text;
    let report = execute(&replay_options)?;
    let expected_energy_bits: u32 = read("final_energy_bits")?.parse()?;
    let expected_checksum: u64 = read("weight_checksum")?.parse()?;
    if report.final_energy.to_bits() != expected_energy_bits
        || report.weight_checksum != expected_checksum
    {
        return Err("deterministic replay mismatch".into());
    }
    println!("replay=verified checksum={:016x}", report.weight_checksum);
    Ok(())
}

fn coordinator() -> Result<ArgonCoordinator, Box<dyn Error>> {
    let grid = PersistentGridConfig {
        grid_dim_x: 1,
        block_dim_x: 64,
        shared_mem_bytes: 16_384,
        tile_dim: TILE_DIM,
    };
    let plasticity = PlasticityConfig {
        learning_rate: 0.002,
        goodness_threshold: 1.0,
        weight_decay: 0.000_5,
        phase_polarity: 1.0,
    };
    Ok(ArgonCoordinator::new(2, &grid, plasticity, 8)?)
}

fn elapsed_nanos(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn percentile(sorted: &[u64], percent: usize) -> u64 {
    let rank = sorted
        .len()
        .saturating_mul(percent)
        .saturating_add(99)
        .saturating_div(100)
        .saturating_sub(1);
    sorted.get(rank).copied().unwrap_or(0)
}
