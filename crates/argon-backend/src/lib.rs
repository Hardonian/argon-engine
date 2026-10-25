//! Explicit backend selection, accelerated GPU execution, and capability reporting.

#![forbid(non_ascii_idents)]

pub mod accelerated;

pub use accelerated::{
    AcceleratedBackendKind, AcceleratedCapabilities, MetalBackend, RocmBackend, VulkanBackend,
};

use argon_runtime::{ArgonCoordinator, RuntimeError, TickTelemetry};
use serde_json::{json, Value};
use std::fmt::{self, Write as FmtWrite};
use std::io::{BufRead, BufReader, Read, Write as IoWrite};
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
    /// Only clear-text local HTTP (or explicitly configured HTTPS) is supported.
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

/// A parsed HTTP endpoint with a local-by-default network policy and optional authentication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalEndpoint {
    host: String,
    port: u16,
    base_path: String,
    is_tls: bool,
    auth_token: Option<String>,
}

impl LocalEndpoint {
    /// Parses an `http://` or `https://` endpoint. Remote hosts require `allow_non_loopback`.
    ///
    /// # Errors
    /// Returns [`LocalStackError`] for malformed endpoints and remote hosts
    /// without explicit opt-in.
    pub fn parse(value: &str, allow_non_loopback: bool) -> Result<Self, LocalStackError> {
        Self::parse_with_auth(value, allow_non_loopback, None)
    }

    /// Parses an endpoint with optional bearer authentication token.
    ///
    /// # Errors
    /// Returns [`LocalStackError`] for malformed endpoints or unauthorized remote addresses.
    pub fn parse_with_auth(
        value: &str,
        allow_non_loopback: bool,
        auth_token: Option<String>,
    ) -> Result<Self, LocalStackError> {
        let (remainder, is_tls) = if let Some(http) = value.strip_prefix("http://") {
            (http, false)
        } else if let Some(https) = value.strip_prefix("https://") {
            (https, true)
        } else {
            return Err(LocalStackError::InvalidEndpoint(
                "endpoint must start with http:// or https://".to_owned(),
            ));
        };
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
        } else if let Some((host, port)) = authority.rsplit_once(':') {
            (
                host.to_owned(),
                port.parse()
                    .map_err(|_| LocalStackError::InvalidEndpoint(value.to_owned()))?,
            )
        } else {
            let default_port = if is_tls { 443 } else { 80 };
            (authority.to_owned(), default_port)
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
            is_tls,
            auth_token,
        })
    }

    /// Returns the endpoint in URL form, suitable for diagnostics.
    pub fn url(&self) -> String {
        let scheme = if self.is_tls { "https" } else { "http" };
        let host = if self.host.contains(':') {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        format!("{scheme}://{host}:{}{}", self.port, self.base_path)
    }

    /// Returns whether this endpoint is configured for TLS.
    pub const fn is_tls(&self) -> bool {
        self.is_tls
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
        let mut headers = format!(
            "{method} {request_path} HTTP/1.0\r\nHost: {}\r\nAccept: application/json\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
            self.host, payload.len()
        );
        if let Some(token) = &self.auth_token {
            let _ = write!(headers, "Authorization: Bearer {token}\r\n");
        }
        headers.push_str("\r\n");
        headers.push_str(payload);

        stream
            .write_all(headers.as_bytes())
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

    fn request_stream<F>(
        &self,
        method: &str,
        path: &str,
        body: Option<&str>,
        kind: LocalStackKind,
        mut on_chunk: F,
    ) -> Result<String, LocalStackError>
    where
        F: FnMut(&str),
    {
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
        let mut headers = format!(
            "{method} {request_path} HTTP/1.0\r\nHost: {}\r\nAccept: text/event-stream, application/x-ndjson, application/json\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
            self.host, payload.len()
        );
        if let Some(token) = &self.auth_token {
            let _ = write!(headers, "Authorization: Bearer {token}\r\n");
        }
        headers.push_str("\r\n");
        headers.push_str(payload);

        stream
            .write_all(headers.as_bytes())
            .map_err(|error| LocalStackError::Transport(error.to_string()))?;

        let mut reader = BufReader::new(stream);
        let mut status_line = String::new();
        reader
            .read_line(&mut status_line)
            .map_err(|error| LocalStackError::Transport(error.to_string()))?;
        let status = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|val| val.parse::<u16>().ok());
        if !matches!(status, Some(200..=299)) {
            return Err(LocalStackError::Protocol(status_line.trim().to_owned()));
        }

        let mut header_line = String::new();
        loop {
            header_line.clear();
            let bytes = reader
                .read_line(&mut header_line)
                .map_err(|error| LocalStackError::Transport(error.to_string()))?;
            if bytes == 0 || header_line == "\r\n" || header_line == "\n" {
                break;
            }
        }

        let mut accumulated = String::new();
        let mut line = String::new();
        loop {
            line.clear();
            let bytes = reader
                .read_line(&mut line)
                .map_err(|error| LocalStackError::Transport(error.to_string()))?;
            if bytes == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let (chunk, is_done) = parse_stream_line(trimmed, kind);
            if let Some(token) = chunk {
                on_chunk(&token);
                accumulated.push_str(&token);
            }
            if is_done {
                break;
            }
        }

        if accumulated.is_empty() {
            return Err(LocalStackError::MissingResponse);
        }
        Ok(accumulated)
    }

    fn host_is_explicitly_remote_allowed(&self) -> bool {
        !matches!(self.host.as_str(), "localhost" | "127.0.0.1" | "::1")
    }
}

fn parse_stream_line(trimmed: &str, kind: LocalStackKind) -> (Option<String>, bool) {
    match kind {
        LocalStackKind::Ollama => {
            if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                let token = val
                    .get("response")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(ToOwned::to_owned);
                let done = val.get("done").and_then(Value::as_bool) == Some(true);
                return (token, done);
            }
        }
        LocalStackKind::LlamaCpp => {
            if let Some(data) = trimmed.strip_prefix("data:") {
                let data = data.trim();
                if data == "[DONE]" {
                    return (None, true);
                }
                if let Ok(val) = serde_json::from_str::<Value>(data) {
                    let token = val
                        .get("content")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .map(ToOwned::to_owned);
                    let stop = val.get("stop").and_then(Value::as_bool) == Some(true);
                    return (token, stop);
                }
            }
        }
        LocalStackKind::LmStudio | LocalStackKind::OpenAiCompatible => {
            if let Some(data) = trimmed.strip_prefix("data:") {
                let data = data.trim();
                if data == "[DONE]" {
                    return (None, true);
                }
                if let Ok(val) = serde_json::from_str::<Value>(data) {
                    let token = val
                        .get("choices")
                        .and_then(Value::as_array)
                        .and_then(|c| c.first())
                        .and_then(|c| c.get("delta"))
                        .and_then(|d| d.get("content"))
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .map(ToOwned::to_owned);
                    return (token, false);
                }
            }
        }
        LocalStackKind::Auto => {}
    }
    (None, false)
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
        Self::new_authenticated(kind, endpoint, allow_non_loopback, None)
    }

    /// Creates an authenticated client with an optional bearer token.
    ///
    /// # Errors
    /// Returns [`LocalStackError`] when an endpoint is missing, malformed, or
    /// violates network routing policy.
    pub fn new_authenticated(
        kind: LocalStackKind,
        endpoint: Option<&str>,
        allow_non_loopback: bool,
        auth_token: Option<String>,
    ) -> Result<Self, LocalStackError> {
        let endpoint = endpoint
            .or_else(|| kind.default_endpoint())
            .ok_or_else(|| {
                LocalStackError::InvalidEndpoint("auto requires discovery".to_owned())
            })?;
        Ok(Self {
            kind,
            endpoint: LocalEndpoint::parse_with_auth(endpoint, allow_non_loopback, auth_token)?,
        })
    }

    /// Attaches an authentication bearer token to this client.
    #[must_use]
    pub fn with_auth(mut self, auth_token: Option<String>) -> Self {
        self.endpoint.auth_token = auth_token;
        self
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
                json!({"prompt": prompt, "n_predict": max_tokens, "stream": false}),
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

    /// Generates a completion while streaming tokens in real time.
    ///
    /// # Errors
    /// Returns [`LocalStackError`] on connection failures or malformed streams.
    pub fn complete_stream<F>(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: u32,
        on_chunk: F,
    ) -> Result<String, LocalStackError>
    where
        F: FnMut(&str),
    {
        let (path, payload) = match self.kind {
            LocalStackKind::Ollama => (
                "/api/generate",
                json!({"model": model, "prompt": prompt, "stream": true, "options": {"num_predict": max_tokens}}),
            ),
            LocalStackKind::LlamaCpp => (
                "/completion",
                json!({"prompt": prompt, "n_predict": max_tokens, "stream": true}),
            ),
            LocalStackKind::LmStudio | LocalStackKind::OpenAiCompatible => (
                "/v1/chat/completions",
                json!({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": true}),
            ),
            LocalStackKind::Auto => return Err(LocalStackError::InvalidStack("auto".to_owned())),
        };
        self.endpoint.request_stream(
            "POST",
            path,
            Some(&payload.to_string()),
            self.kind,
            on_chunk,
        )
    }

    /// Triggers a model pull or download on the server.
    ///
    /// # Errors
    /// Returns [`LocalStackError`] if the request fails or is unsupported.
    pub fn pull_model(&self, model: &str) -> Result<(), LocalStackError> {
        match self.kind {
            LocalStackKind::Ollama => {
                let payload = json!({"name": model, "stream": false});
                let _ = self
                    .endpoint
                    .request("POST", "/api/pull", Some(&payload.to_string()))?;
                Ok(())
            }
            LocalStackKind::LlamaCpp
            | LocalStackKind::LmStudio
            | LocalStackKind::OpenAiCompatible => {
                let models = self.list_models()?;
                if models.iter().any(|item| item == model) {
                    Ok(())
                } else {
                    Err(LocalStackError::Protocol(format!(
                        "model '{model}' not found on server"
                    )))
                }
            }
            LocalStackKind::Auto => Err(LocalStackError::InvalidStack("auto".to_owned())),
        }
    }

    /// Evicts or unloads a model from resident memory / VRAM.
    ///
    /// # Errors
    /// Returns [`LocalStackError`] if the unload request fails.
    pub fn unload_model(&self, model: &str) -> Result<(), LocalStackError> {
        match self.kind {
            LocalStackKind::Ollama => {
                let payload = json!({"model": model, "keep_alive": 0});
                let _ =
                    self.endpoint
                        .request("POST", "/api/generate", Some(&payload.to_string()))?;
                Ok(())
            }
            LocalStackKind::LlamaCpp
            | LocalStackKind::LmStudio
            | LocalStackKind::OpenAiCompatible => Ok(()),
            LocalStackKind::Auto => Err(LocalStackError::InvalidStack("auto".to_owned())),
        }
    }

    /// Retrieves detailed model metadata and inspection attributes.
    ///
    /// # Errors
    /// Returns [`LocalStackError`] if metadata retrieval fails.
    pub fn model_info(&self, model: &str) -> Result<Value, LocalStackError> {
        match self.kind {
            LocalStackKind::Ollama => {
                let payload = json!({"name": model});
                self.endpoint
                    .request("POST", "/api/show", Some(&payload.to_string()))
            }
            LocalStackKind::LlamaCpp
            | LocalStackKind::LmStudio
            | LocalStackKind::OpenAiCompatible => {
                let path = format!("/v1/models/{model}");
                if let Ok(info) = self.endpoint.request("GET", &path, None) {
                    Ok(info)
                } else {
                    let list = self.endpoint.request("GET", "/v1/models", None)?;
                    if let Some(items) = list.get("data").and_then(Value::as_array) {
                        if let Some(found) = items
                            .iter()
                            .find(|item| item.get("id").and_then(Value::as_str) == Some(model))
                        {
                            return Ok(found.clone());
                        }
                    }
                    Err(LocalStackError::Protocol(format!(
                        "model '{model}' not found on server"
                    )))
                }
            }
            LocalStackKind::Auto => Err(LocalStackError::InvalidStack("auto".to_owned())),
        }
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
    use argon_kernel::{KernelError, PersistentGridConfig, PlasticityConfig};
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
    fn vulkan_backend_matches_golden_vector() -> Result<(), KernelError> {
        let mut weights = vec![0.0_f32; 16];
        for i in 0..4 {
            weights[i * 4 + i] = 0.5;
        }
        let plasticity = PlasticityConfig {
            learning_rate: 0.01,
            goodness_threshold: 1.0,
            weight_decay: 0.001,
            phase_polarity: 1.0,
        };
        let mut backend = VulkanBackend::new(4, &weights, plasticity)?;
        let mut output = [0.0_f32; 4];
        let input = [0.1, 0.2, 0.3, 0.4];

        let telemetry = backend
            .step(101, 1.0, &input, &mut output)
            .map_err(|_| KernelError::InvalidHyperparameter("backend step failed"))?;

        assert_eq!(telemetry.sequence_id, 101);
        assert!((telemetry.mean_goodness - 0.018_75).abs() < 1e-7);

        let golden = [0.365_145_7, 0.730_291_4, 1.095_437_2, 1.460_582_9];
        for (&actual, &expected) in output.iter().zip(&golden) {
            assert!((actual - expected).abs() < 2e-6);
        }
        assert!((backend.weights()[0] - 0.499_536_37).abs() < 2e-7);
        assert!((backend.weights()[1] - 0.000_072_735_62).abs() < 2e-9);
        Ok(())
    }

    #[test]
    fn metal_and_rocm_backends_match_golden_vector() -> Result<(), KernelError> {
        let mut weights = vec![0.0_f32; 16];
        for i in 0..4 {
            weights[i * 4 + i] = 0.5;
        }
        let plasticity = PlasticityConfig {
            learning_rate: 0.01,
            goodness_threshold: 1.0,
            weight_decay: 0.001,
            phase_polarity: 1.0,
        };

        // Metal backend
        let mut metal = MetalBackend::new(4, &weights, plasticity)?;
        let mut out_metal = [0.0_f32; 4];
        let t_metal = metal
            .step(201, 1.0, &[0.1, 0.2, 0.3, 0.4], &mut out_metal)
            .map_err(|_| KernelError::InvalidHyperparameter("metal step failed"))?;
        assert!((t_metal.mean_goodness - 0.018_75).abs() < 1e-7);

        // ROCm backend
        let mut rocm = RocmBackend::new(4, &weights, plasticity)?;
        let mut out_rocm = [0.0_f32; 4];
        let t_rocm = rocm
            .step(301, 1.0, &[0.1, 0.2, 0.3, 0.4], &mut out_rocm)
            .map_err(|_| KernelError::InvalidHyperparameter("rocm step failed"))?;
        assert!((t_rocm.mean_goodness - 0.018_75).abs() < 1e-7);

        // Check output parity between all backends
        for (m, r) in out_metal.iter().zip(&out_rocm) {
            assert!((m - r).abs() < 1e-7);
        }
        Ok(())
    }

    #[test]
    fn accelerated_backends_stable_over_1000_ticks() -> Result<(), KernelError> {
        let weights = vec![0.25_f32; 64];
        let plasticity = PlasticityConfig {
            learning_rate: 0.02,
            goodness_threshold: 2.0,
            weight_decay: 0.0005,
            phase_polarity: 1.0,
        };
        let mut vk = VulkanBackend::new(8, &weights, plasticity)?;
        let input = [0.5_f32, 0.3, 0.8, 0.1, 0.6, 0.4, 0.7, 0.2];
        let mut output = [0.0_f32; 8];

        for step in 0_u64..1_000_u64 {
            let polarity = if step % 2 == 0 { 1.0 } else { -1.0 };
            let t = vk
                .step(step, polarity, &input, &mut output)
                .map_err(|_| KernelError::InvalidHyperparameter("step error"))?;
            assert!(t.mean_goodness.is_finite());
            assert!(t.output_l2_norm.is_finite());
        }
        assert_eq!(vk.cumulative_steps(), 1_000);
        assert!(vk.weights().iter().all(|w| w.is_finite()));
        Ok(())
    }

    #[test]
    fn accelerated_capabilities_probe_runs() {
        let caps = AcceleratedCapabilities::probe();
        assert!(!caps.detected_platforms.is_empty() || !caps.vulkan_available);
    }

    #[test]
    fn local_endpoints_are_loopback_by_default() -> Result<(), LocalStackError> {
        let endpoint = LocalEndpoint::parse("http://127.0.0.1:11434/api", false)?;
        assert_eq!(endpoint.url(), "http://127.0.0.1:11434/api");
        assert!(!endpoint.is_tls());

        let tls_endpoint = LocalEndpoint::parse("https://127.0.0.1:8443", false)?;
        assert_eq!(tls_endpoint.url(), "https://127.0.0.1:8443");
        assert!(tls_endpoint.is_tls());

        assert!(matches!(
            LocalEndpoint::parse("http://example.com:8080", false),
            Err(LocalStackError::NonLocalEndpoint(_))
        ));
        assert!(matches!(
            LocalEndpoint::parse("ftp://127.0.0.1:11434", false),
            Err(LocalStackError::InvalidEndpoint(_))
        ));
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
    fn streaming_and_model_lifecycle_for_openai_and_ollama() -> Result<(), LocalStackError> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .map_err(|error| LocalStackError::Transport(error.to_string()))?;
        let port = listener
            .local_addr()
            .map_err(|error| LocalStackError::Transport(error.to_string()))?
            .port();

        let server = thread::spawn(move || -> Result<(), LocalStackError> {
            for _ in 0..6 {
                let (mut stream, _) = listener
                    .accept()
                    .map_err(|error| LocalStackError::Transport(error.to_string()))?;
                let mut request = [0_u8; 8_192];
                let bytes = stream
                    .read(&mut request)
                    .map_err(|error| LocalStackError::Transport(error.to_string()))?;
                let req_str = String::from_utf8_lossy(&request[..bytes]);

                let (headers, body) = req_str.split_once("\r\n\r\n").unwrap_or((&req_str, ""));
                let is_auth_ok = headers.contains("Authorization: Bearer test-key");

                let response = if req_str.starts_with("GET /v1/models ") {
                    "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"data\":[{\"id\":\"local-model-v2\"}]}".to_owned()
                } else if req_str.starts_with("POST /v1/chat/completions ")
                    && req_str.contains("\"stream\":true")
                {
                    "HTTP/1.0 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"streamed \"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"output\"}}]}\n\ndata: [DONE]\n\n".to_owned()
                } else if req_str.starts_with("POST /api/pull ") {
                    "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: 17\r\n\r\n{\"status\":\"pulled\"}".to_owned()
                } else if req_str.starts_with("POST /api/generate ")
                    && body.contains("\"keep_alive\":0")
                {
                    "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: 19\r\n\r\n{\"status\":\"unloaded\"}".to_owned()
                } else if req_str.starts_with("POST /api/show ") {
                    "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: 42\r\n\r\n{\"details\":{\"format\":\"gguf\",\"family\":\"qwen\"}}".to_owned()
                } else if is_auth_ok {
                    "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\n\r\n{\"auth\":\"valid\"}".to_owned()
                } else {
                    "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}".to_owned()
                };

                stream
                    .write_all(response.as_bytes())
                    .map_err(|error| LocalStackError::Transport(error.to_string()))?;
            }
            Ok(())
        });

        let stack = LocalAiStack::new_authenticated(
            LocalStackKind::OpenAiCompatible,
            Some(&format!("http://127.0.0.1:{port}")),
            false,
            Some("test-key".to_owned()),
        )?;

        // 1. Health & list models
        stack.health_check()?;
        let models = stack.list_models()?;
        assert_eq!(models, ["local-model-v2"]);

        // 2. Stream completions
        let mut chunks = Vec::new();
        let full_text = stack.complete_stream("local-model-v2", "test prompt", 32, |chunk| {
            chunks.push(chunk.to_owned());
        })?;
        assert_eq!(full_text, "streamed output");
        assert_eq!(chunks, vec!["streamed ", "output"]);

        // 3. Ollama lifecycle endpoints
        let ollama_stack = LocalAiStack::new(
            LocalStackKind::Ollama,
            Some(&format!("http://127.0.0.1:{port}")),
            false,
        )?;
        ollama_stack.pull_model("qwen-3")?;
        ollama_stack.unload_model("qwen-3")?;
        let info = ollama_stack.model_info("qwen-3")?;
        assert_eq!(info["details"]["format"], "gguf");

        server
            .join()
            .map_err(|_| LocalStackError::Transport("test server panicked".to_owned()))??;
        Ok(())
    }
}
