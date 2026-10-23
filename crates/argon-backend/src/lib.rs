//! Explicit backend selection and capability reporting.

#![forbid(non_ascii_idents)]

use argon_runtime::{ArgonCoordinator, RuntimeError, TickTelemetry};
use serde_json::{json, Value};
use std::fmt;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::process::Command;
use std::time::Duration;

/// Execution engine contract shared by synchronous backends.
pub trait ExecutionBackend {
    /// Returns the backend's required feature dimension.
    fn dimension(&self) -> usize;

    /// Executes one borrowed-buffer tick.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] for invalid inputs or backend execution errors.
    fn step(
        &mut self,
        sequence_id: u64,
        polarity: f32,
        features: &[f32],
        output: &mut [f32],
    ) -> Result<TickTelemetry, RuntimeError>;
}

/// Normative scalar/portable-SIMD CPU backend.
pub struct CpuBackend {
    coordinator: ArgonCoordinator,
}

impl CpuBackend {
    /// Wraps an initialized coordinator.
    pub const fn new(coordinator: ArgonCoordinator) -> Self {
        Self { coordinator }
    }

    /// Returns the wrapped coordinator for checkpointing and telemetry.
    pub const fn coordinator(&self) -> &ArgonCoordinator {
        &self.coordinator
    }
}

impl ExecutionBackend for CpuBackend {
    fn dimension(&self) -> usize {
        self.coordinator.tile_dim()
    }

    fn step(
        &mut self,
        sequence_id: u64,
        polarity: f32,
        features: &[f32],
        output: &mut [f32],
    ) -> Result<TickTelemetry, RuntimeError> {
        self.coordinator
            .step_borrowed(sequence_id, polarity, features, output)
    }
}

/// CUDA driver and device information discovered without selecting the backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaCapabilities {
    /// NVIDIA driver version reported by `nvidia-smi`.
    pub driver_version: String,
    /// Names of visible CUDA devices.
    pub devices: Vec<String>,
}

/// A locally hosted model-server API supported by Argon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalStackKind {
    /// Discover a supported server at its conventional loopback address.
    Auto,
    /// Ollama's local `/api/generate` API.
    Ollama,
    /// llama.cpp's OpenAI-compatible `/completion` API.
    LlamaCpp,
    /// LM Studio's OpenAI-compatible local server.
    LmStudio,
    /// A self-hosted OpenAI-compatible `/v1/chat/completions` API.
    OpenAiCompatible,
}

impl LocalStackKind {
    /// Parses a stable CLI/config spelling.
    ///
    /// # Errors
    /// Returns [`LocalStackError::InvalidStack`] when the spelling is unsupported.
    pub fn parse(value: &str) -> Result<Self, LocalStackError> {
        match value {
            "auto" => Ok(Self::Auto),
            "ollama" => Ok(Self::Ollama),
            "llama.cpp" | "llama-cpp" | "llamacpp" => Ok(Self::LlamaCpp),
            "lm-studio" | "lmstudio" => Ok(Self::LmStudio),
            "openai-compatible" | "openai" => Ok(Self::OpenAiCompatible),
            _ => Err(LocalStackError::InvalidStack(value.to_owned())),
        }
    }

    /// Conventional local endpoint for this server.
    pub const fn default_endpoint(self) -> Option<&'static str> {
        match self {
            Self::Auto => None,
            Self::Ollama => Some("http://127.0.0.1:11434"),
            Self::LlamaCpp => Some("http://127.0.0.1:8080"),
            Self::LmStudio => Some("http://127.0.0.1:1234"),
            Self::OpenAiCompatible => Some("http://127.0.0.1:8000"),
        }
    }
}

/// Failure while configuring or talking to a local model server.
#[derive(Debug)]
pub enum LocalStackError {
    /// The requested stack name is not supported.
    InvalidStack(String),
    /// Only clear-text local HTTP is intentionally supported by the built-in client.
    InvalidEndpoint(String),
    /// A remote endpoint was attempted without explicit opt-in.
    NonLocalEndpoint(String),
    /// Network transport failed.
    Transport(String),
    /// The local server returned an invalid or failing HTTP response.
    Protocol(String),
    /// The response did not contain the requested generation field.
    MissingResponse,
}

impl fmt::Display for LocalStackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStack(value) => write!(formatter, "unsupported local stack: {value}"),
            Self::InvalidEndpoint(value) => write!(formatter, "invalid local endpoint: {value}"),
            Self::NonLocalEndpoint(value) => write!(formatter, "endpoint is not loopback: {value}"),
            Self::Transport(value) => write!(formatter, "local stack transport failed: {value}"),
            Self::Protocol(value) => write!(formatter, "local stack protocol failed: {value}"),
            Self::MissingResponse => {
                write!(formatter, "local stack response omitted generated text")
            }
        }
    }
}

impl std::error::Error for LocalStackError {}

/// A parsed HTTP endpoint with a local-by-default network policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalEndpoint {
    host: String,
    port: u16,
    base_path: String,
}

impl LocalEndpoint {
    /// Parses an `http://` endpoint. Remote hosts require `allow_non_loopback`.
    ///
    /// # Errors
    /// Returns [`LocalStackError`] for malformed endpoints and remote hosts
    /// without explicit opt-in.
    pub fn parse(value: &str, allow_non_loopback: bool) -> Result<Self, LocalStackError> {
        let remainder = value.strip_prefix("http://").ok_or_else(|| {
            LocalStackError::InvalidEndpoint("only http:// endpoints are supported".to_owned())
        })?;
        let (authority, suffix) = remainder.split_once('/').unwrap_or((remainder, ""));
        let (host, port) = if let Some(bracketed) = authority.strip_prefix('[') {
            let (host, port) = bracketed.split_once("]:").ok_or_else(|| {
                LocalStackError::InvalidEndpoint("IPv6 addresses require [address]:port".to_owned())
            })?;
            (
                host.to_owned(),
                port.parse()
                    .map_err(|_| LocalStackError::InvalidEndpoint(value.to_owned()))?,
            )
        } else {
            let (host, port) = authority.rsplit_once(':').ok_or_else(|| {
                LocalStackError::InvalidEndpoint("endpoint must include a port".to_owned())
            })?;
            (
                host.to_owned(),
                port.parse()
                    .map_err(|_| LocalStackError::InvalidEndpoint(value.to_owned()))?,
            )
        };
        let is_loopback = matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1");
        if !is_loopback && !allow_non_loopback {
            return Err(LocalStackError::NonLocalEndpoint(host));
        }
        let base_path = if suffix.is_empty() {
            String::new()
        } else {
            format!("/{suffix}")
        };
        Ok(Self {
            host,
            port,
            base_path,
        })
    }

    /// Returns the endpoint in URL form, suitable for diagnostics.
    pub fn url(&self) -> String {
        let host = if self.host.contains(':') {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        format!("http://{host}:{}{}", self.port, self.base_path)
    }

    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<&str>,
    ) -> Result<Value, LocalStackError> {
        let address = format!("{}:{}", self.host, self.port);
        let socket = address
            .to_socket_addrs()
            .map_err(|error| LocalStackError::Transport(error.to_string()))?
            .next()
            .ok_or_else(|| LocalStackError::Transport("endpoint did not resolve".to_owned()))?;
        if !socket.ip().is_loopback() && !self.host_is_explicitly_remote_allowed() {
            return Err(LocalStackError::NonLocalEndpoint(socket.ip().to_string()));
        }
        let mut stream = TcpStream::connect_timeout(&socket, Duration::from_secs(3))
            .map_err(|error| LocalStackError::Transport(error.to_string()))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(60)))
            .map_err(|error| LocalStackError::Transport(error.to_string()))?;
        let request_path = format!("{}{}", self.base_path, path);
        let payload = body.unwrap_or("");
        let request = format!(
            "{method} {request_path} HTTP/1.0\r\nHost: {}\r\nAccept: application/json\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
            self.host, payload.len()
        );
        stream
            .write_all(request.as_bytes())
            .map_err(|error| LocalStackError::Transport(error.to_string()))?;
        let mut response = String::new();
        stream
            .take(16 * 1024 * 1024)
            .read_to_string(&mut response)
            .map_err(|error| LocalStackError::Transport(error.to_string()))?;
        let (head, body) = response
            .split_once("\r\n\r\n")
            .ok_or_else(|| LocalStackError::Protocol("malformed HTTP response".to_owned()))?;
        let status = head
            .split_whitespace()
            .nth(1)
            .and_then(|value| value.parse::<u16>().ok());
        if !matches!(status, Some(200..=299)) {
            return Err(LocalStackError::Protocol(
                head.lines()
                    .next()
                    .unwrap_or("unknown HTTP status")
                    .to_owned(),
            ));
        }
        serde_json::from_str(body).map_err(|error| LocalStackError::Protocol(error.to_string()))
    }

    fn host_is_explicitly_remote_allowed(&self) -> bool {
        !matches!(self.host.as_str(), "localhost" | "127.0.0.1" | "::1")
    }
}

/// Local, sovereign model-server client. It never contacts a remote host unless
/// an application explicitly opts into a non-loopback endpoint.
#[derive(Debug, Clone)]
pub struct LocalAiStack {
    /// Server protocol used by this client.
    pub kind: LocalStackKind,
    /// Endpoint used by this client.
    pub endpoint: LocalEndpoint,
}

impl LocalAiStack {
    /// Creates a client. `None` selects the stack's conventional loopback endpoint.
    ///
    /// # Errors
    /// Returns [`LocalStackError`] when an endpoint is missing, malformed, or
    /// violates the local-only policy.
    pub fn new(
        kind: LocalStackKind,
        endpoint: Option<&str>,
        allow_non_loopback: bool,
    ) -> Result<Self, LocalStackError> {
        let endpoint = endpoint
            .or_else(|| kind.default_endpoint())
            .ok_or_else(|| {
                LocalStackError::InvalidEndpoint("auto requires discovery".to_owned())
            })?;
        Ok(Self {
            kind,
            endpoint: LocalEndpoint::parse(endpoint, allow_non_loopback)?,
        })
    }

    /// Finds a responding server without starting processes or sending prompts.
    pub fn discover() -> Option<Self> {
        for kind in [
            LocalStackKind::Ollama,
            LocalStackKind::LlamaCpp,
            LocalStackKind::LmStudio,
            LocalStackKind::OpenAiCompatible,
        ] {
            let stack = Self::new(kind, None, false).ok()?;
            if stack.health_check().is_ok() {
                return Some(stack);
            }
        }
        None
    }

    /// Confirms that the expected local API responds.
    ///
    /// # Errors
    /// Returns [`LocalStackError`] if the server is unavailable or does not
    /// return a successful JSON response.
    pub fn health_check(&self) -> Result<(), LocalStackError> {
        let path = match self.kind {
            LocalStackKind::Ollama => "/api/tags",
            LocalStackKind::LlamaCpp
            | LocalStackKind::LmStudio
            | LocalStackKind::OpenAiCompatible => "/v1/models",
            LocalStackKind::Auto => return Err(LocalStackError::InvalidStack("auto".to_owned())),
        };
        let _response = self.endpoint.request("GET", path, None)?;
        Ok(())
    }

    /// Lists installed or served model identifiers without generating text.
    ///
    /// # Errors
    /// Returns [`LocalStackError`] when the local server cannot be reached or
    /// its model-list response is not recognized.
    pub fn list_models(&self) -> Result<Vec<String>, LocalStackError> {
        let response = match self.kind {
            LocalStackKind::Ollama => self.endpoint.request("GET", "/api/tags", None)?,
            LocalStackKind::LlamaCpp
            | LocalStackKind::LmStudio
            | LocalStackKind::OpenAiCompatible => {
                self.endpoint.request("GET", "/v1/models", None)?
            }
            LocalStackKind::Auto => return Err(LocalStackError::InvalidStack("auto".to_owned())),
        };
        let models = if self.kind == LocalStackKind::Ollama {
            response
                .get("models")
                .and_then(Value::as_array)
                .map(|models| {
                    models
                        .iter()
                        .filter_map(|model| {
                            model
                                .get("name")
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned)
                        })
                        .collect()
                })
        } else {
            response
                .get("data")
                .and_then(Value::as_array)
                .map(|models| {
                    models
                        .iter()
                        .filter_map(|model| {
                            model
                                .get("id")
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned)
                        })
                        .collect()
                })
        };
        models.ok_or_else(|| {
            LocalStackError::Protocol("model list has an unexpected JSON shape".to_owned())
        })
    }

    /// Generates a completion using the local stack's native HTTP API.
    ///
    /// # Errors
    /// Returns [`LocalStackError`] for connection failures, failed API
    /// responses, or responses without generated text.
    pub fn complete(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<String, LocalStackError> {
        let (path, payload) = match self.kind {
            LocalStackKind::Ollama => (
                "/api/generate",
                json!({"model": model, "prompt": prompt, "stream": false, "options": {"num_predict": max_tokens}}),
            ),
            LocalStackKind::LlamaCpp => (
                "/completion",
                json!({"prompt": prompt, "n_predict": max_tokens}),
            ),
            LocalStackKind::LmStudio | LocalStackKind::OpenAiCompatible => (
                "/v1/chat/completions",
                json!({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": false}),
            ),
            LocalStackKind::Auto => return Err(LocalStackError::InvalidStack("auto".to_owned())),
        };
        let response = self
            .endpoint
            .request("POST", path, Some(&payload.to_string()))?;
        let text = match self.kind {
            LocalStackKind::Ollama => response.get("response").and_then(Value::as_str),
            LocalStackKind::LlamaCpp => response.get("content").and_then(Value::as_str),
            LocalStackKind::LmStudio | LocalStackKind::OpenAiCompatible => response
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("message"))
                .and_then(|message| message.get("content"))
                .and_then(Value::as_str),
            LocalStackKind::Auto => None,
        };
        text.map(ToOwned::to_owned)
            .ok_or(LocalStackError::MissingResponse)
    }
}

impl CudaCapabilities {
    /// Probes the local NVIDIA management interface.
    ///
    /// Absence is represented by `None` and never silently falls back while
    /// claiming CUDA execution.
    pub fn probe() -> Option<Self> {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=driver_version,name",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = String::from_utf8(output.stdout).ok()?;
        let mut driver_version = None;
        let mut devices = Vec::new();
        for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
            let (driver, name) = line.split_once(',')?;
            driver_version.get_or_insert_with(|| driver.trim().to_owned());
            devices.push(name.trim().to_owned());
        }
        Some(Self {
            driver_version: driver_version?,
            devices,
        })
    }
}

/// GPU class used for local-model sizing and user-facing diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsumerGpuTier {
    /// Less than 6 GiB VRAM: small quantized models and CPU offload.
    Entry,
    /// 6–11 GiB VRAM: 7B/8B quantized models.
    Mainstream,
    /// 12–23 GiB VRAM: comfortable 8B–14B quantized models.
    Enthusiast,
    /// At least 24 GiB VRAM: larger quantized local models.
    HighMemory,
}

impl ConsumerGpuTier {
    /// Human-readable, conservative local-model guidance.
    pub const fn guidance(self) -> &'static str {
        match self {
            Self::Entry => "small quantized models; use CPU offload for larger models",
            Self::Mainstream => "7B/8B quantized models",
            Self::Enthusiast => "8B–14B quantized models with useful context",
            Self::HighMemory => "larger quantized models and longer context windows",
        }
    }
}

/// A locally visible GPU. Discovery is informational and never changes backend selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDevice {
    /// Vendor-reported name.
    pub name: String,
    /// Installed VRAM in MiB when the driver reports it.
    pub memory_mib: Option<u32>,
    /// Driver version when the driver reports it.
    pub driver_version: Option<String>,
    /// The applicable local-model capacity tier.
    pub tier: ConsumerGpuTier,
}

impl GpuDevice {
    fn with_memory(name: String, memory_mib: Option<u32>, driver_version: Option<String>) -> Self {
        let tier = match memory_mib.unwrap_or(0) {
            0..=6_143 => ConsumerGpuTier::Entry,
            6_144..=12_287 => ConsumerGpuTier::Mainstream,
            12_288..=24_575 => ConsumerGpuTier::Enthusiast,
            _ => ConsumerGpuTier::HighMemory,
        };
        Self {
            name,
            memory_mib,
            driver_version,
            tier,
        }
    }
}

/// Hardware inventory for local-first deployment planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalHardware {
    /// CUDA devices exposed by the NVIDIA driver.
    pub gpus: Vec<GpuDevice>,
    /// Whether the portable CPU runtime is always available.
    pub cpu_available: bool,
}

impl LocalHardware {
    /// Probes consumer GPU drivers without selecting an accelerated backend.
    pub fn probe() -> Self {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ])
            .output();
        let gpus = output
            .ok()
            .filter(|value| value.status.success())
            .map(|value| {
                String::from_utf8_lossy(&value.stdout)
                    .lines()
                    .filter_map(|line| {
                        let mut columns = line.split(',').map(str::trim);
                        let name = columns.next()?.to_owned();
                        let memory_mib = columns.next().and_then(|value| value.parse().ok());
                        let driver_version = columns.next().map(ToOwned::to_owned);
                        Some(GpuDevice::with_memory(name, memory_mib, driver_version))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            gpus,
            cpu_available: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argon_kernel::{PersistentGridConfig, PlasticityConfig};
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn cpu_backend_executes_borrowed_tick() -> Result<(), RuntimeError> {
        let grid = PersistentGridConfig {
            grid_dim_x: 1,
            block_dim_x: 32,
            shared_mem_bytes: 4_096,
            tile_dim: 4,
        };
        let plasticity = PlasticityConfig {
            learning_rate: 0.01,
            goodness_threshold: 1.0,
            weight_decay: 0.001,
            phase_polarity: 1.0,
        };
        let coordinator = ArgonCoordinator::new(1, &grid, plasticity, 4)?;
        let mut backend = CpuBackend::new(coordinator);
        let mut output = [0.0; 4];
        let telemetry = backend.step(9, 1.0, &[0.1, 0.2, 0.3, 0.4], &mut output)?;
        assert_eq!(telemetry.sequence_id, 9);
        assert!(output.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn local_endpoints_are_loopback_by_default() -> Result<(), LocalStackError> {
        let endpoint = LocalEndpoint::parse("http://127.0.0.1:11434/api", false)?;
        assert_eq!(endpoint.url(), "http://127.0.0.1:11434/api");
        assert!(matches!(
            LocalEndpoint::parse("http://example.com:8080", false),
            Err(LocalStackError::NonLocalEndpoint(_))
        ));
        assert!(LocalEndpoint::parse("https://127.0.0.1:11434", false).is_err());
        Ok(())
    }

    #[test]
    fn consumer_gpu_tiers_are_conservative() {
        assert_eq!(
            GpuDevice::with_memory("8 GB".to_owned(), Some(8_192), None).tier,
            ConsumerGpuTier::Mainstream
        );
        assert_eq!(
            GpuDevice::with_memory("24 GB".to_owned(), Some(24_576), None).tier,
            ConsumerGpuTier::HighMemory
        );
    }

    #[test]
    fn openai_compatible_stack_lists_models_and_completes() -> Result<(), LocalStackError> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .map_err(|error| LocalStackError::Transport(error.to_string()))?;
        let port = listener
            .local_addr()
            .map_err(|error| LocalStackError::Transport(error.to_string()))?
            .port();
        let server = thread::spawn(move || -> Result<(), LocalStackError> {
            for _ in 0..3 {
                let (mut stream, _) = listener
                    .accept()
                    .map_err(|error| LocalStackError::Transport(error.to_string()))?;
                let mut request = [0_u8; 8_192];
                let bytes = stream
                    .read(&mut request)
                    .map_err(|error| LocalStackError::Transport(error.to_string()))?;
                let request = String::from_utf8_lossy(&request[..bytes]);
                let body = if request.starts_with("GET /v1/models ") {
                    r#"{"data":[{"id":"local-test-model"}]}"#
                } else if request.starts_with("POST /v1/chat/completions ") {
                    r#"{"choices":[{"message":{"content":"local response"}}]}"#
                } else {
                    r#"{"error":"unexpected route"}"#
                };
                let response = format!(
                    "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                );
                stream
                    .write_all(response.as_bytes())
                    .map_err(|error| LocalStackError::Transport(error.to_string()))?;
            }
            Ok(())
        });
        let stack = LocalAiStack::new(
            LocalStackKind::OpenAiCompatible,
            Some(&format!("http://127.0.0.1:{port}")),
            false,
        )?;
        stack.health_check()?;
        assert_eq!(stack.list_models()?, ["local-test-model"]);
        assert_eq!(
            stack.complete("local-test-model", "hello", 16)?,
            "local response"
        );
        server
            .join()
            .map_err(|_| LocalStackError::Transport("test server panicked".to_owned()))??;
        Ok(())
    }
}
